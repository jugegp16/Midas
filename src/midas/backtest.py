"""Backtest engine — replays historical data through strategies."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from midas.allocator import Allocator
from midas.models import (
    AllocationConstraints,
    Direction,
    HoldingPeriod,
    MechanicalIntent,
    Order,
    PortfolioConfig,
    TradeRecord,
)
from midas.rebalancer import Rebalancer
from midas.restrictions import RestrictionTracker
from midas.strategies.base import Strategy, max_warmup


@dataclass
class StrategyStats:
    """Per-strategy performance breakdown."""

    name: str
    trades: int
    buys: int
    sells: int
    win_rate: float  # fraction of profitable sells
    pnl: float  # total realized P&L from sells


@dataclass
class BacktestResult:
    trades: list[TradeRecord]
    final_value: float
    starting_value: float
    buy_and_hold_value: float
    train_trades: list[TradeRecord]
    test_trades: list[TradeRecord]
    train_return: float
    test_return: float
    train_bh_return: float
    test_bh_return: float
    split_date: date | None
    twr: float  # time-weighted return (accounts for cash infusions)
    # New metrics
    equity_curve: list[tuple[date, float]]  # daily (date, portfolio_value)
    cagr: float  # compound annual growth rate
    max_drawdown: float  # peak-to-trough percentage decline
    sharpe_ratio: float  # annualized, risk-free=0
    sortino_ratio: float  # annualized, downside deviation only
    win_rate: float  # fraction of round-trip sells that were profitable
    profit_factor: float  # gross wins / gross losses (inf if no losses)
    avg_win: float  # average P&L of winning sells
    avg_loss: float  # average P&L of losing sells
    efficiency_ratio: float  # test_return / train_return (0 if no split)
    strategy_stats: list[StrategyStats]


@dataclass
class _TickerIndex:
    """Pre-computed numpy arrays and pointer for a single ticker's price data."""

    dates: list[date]
    values: np.ndarray
    ptr: int = 0


@dataclass
class _SimState:
    """Mutable simulation state carried across trading days."""

    positions: dict[str, float] = field(default_factory=dict)
    cost_basis: dict[str, float] = field(default_factory=dict)
    # Cost basis recorded at the moment each SELL trade was executed, in the
    # same order as SELL entries in `trades`. A parallel list (rather than a
    # (date, ticker)-keyed dict) is required so that multiple sells of the
    # same ticker on the same day — e.g. from different strategies — each get
    # their own basis snapshot instead of overwriting one another.
    basis_per_sell: list[float] = field(default_factory=list)
    bh_positions: dict[str, float] = field(default_factory=dict)
    purchase_dates: dict[str, list[tuple[float, date]]] = field(default_factory=dict)
    cash: float = 0.0
    starting_value: float = 0.0
    trades: list[TradeRecord] = field(default_factory=list)
    last_day: date | None = None
    split_value: float | None = None
    split_bh_value: float | None = None
    restriction_tracker: RestrictionTracker | None = None
    twr_base_value: float = 0.0  # portfolio value after last cash infusion
    twr_periods: list[float] = field(default_factory=list)  # sub-period returns
    twr_split_idx: int | None = None  # index into twr_periods at train/test split
    equity_curve: list[tuple[date, float]] = field(default_factory=list)


TRADING_DAYS_PER_YEAR = 252


def compute_cagr(starting: float, final: float, days: int) -> float:
    """Compound annual growth rate from total return over *days* calendar days."""
    if starting <= 0 or final <= 0 or days <= 0:
        return 0.0
    years = days / 365.25
    return float((final / starting) ** (1.0 / years) - 1.0)


def compute_max_drawdown(equity_curve: list[tuple[date, float]]) -> float:
    """Maximum peak-to-trough percentage decline."""
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0][1]
    max_dd = 0.0
    for _, value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def compute_sharpe(equity_curve: list[tuple[date, float]]) -> float:
    """Annualized Sharpe ratio (risk-free = 0) from daily returns."""
    if len(equity_curve) < 3:
        return 0.0
    values = [v for _, v in equity_curve]
    daily_returns = [(values[i] - values[i - 1]) / values[i - 1] for i in range(1, len(values)) if values[i - 1] > 0]
    if len(daily_returns) < 2:
        return 0.0
    mean_r = sum(daily_returns) / len(daily_returns)
    variance = sum((r - mean_r) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
    std_r = math.sqrt(variance) if variance > 0 else 0.0
    if std_r == 0:
        return 0.0
    return (mean_r / std_r) * math.sqrt(TRADING_DAYS_PER_YEAR)


def compute_sortino(equity_curve: list[tuple[date, float]]) -> float:
    """Annualized Sortino ratio (risk-free = 0, downside deviation only).

    When there is no downside (no negative daily returns) the metric is
    undefined. We return 0.0 as a sentinel rather than `inf` so that callers
    averaging across multiple windows (e.g. walk-forward folds) aren't
    poisoned by a single zero-downside fold.
    """
    if len(equity_curve) < 3:
        return 0.0
    values = [v for _, v in equity_curve]
    daily_returns = [(values[i] - values[i - 1]) / values[i - 1] for i in range(1, len(values)) if values[i - 1] > 0]
    if len(daily_returns) < 2:
        return 0.0
    mean_r = sum(daily_returns) / len(daily_returns)
    downside = [r for r in daily_returns if r < 0]
    if not downside:
        return 0.0
    downside_var = sum(r**2 for r in downside) / len(daily_returns)
    downside_dev = math.sqrt(downside_var) if downside_var > 0 else 0.0
    if downside_dev == 0:
        return 0.0
    return (mean_r / downside_dev) * math.sqrt(TRADING_DAYS_PER_YEAR)


def _pair_sells_with_basis(
    trades: list[TradeRecord],
    basis_per_sell: list[float],
) -> list[tuple[TradeRecord, float]]:
    """Zip SELL trades with their recorded cost basis (parallel-list order).

    Falls back to the trade price (zero P&L) for any sell beyond the recorded
    basis list — defensive only; the lists should always be the same length.
    """
    sells = [t for t in trades if t.direction == Direction.SELL]
    paired: list[tuple[TradeRecord, float]] = []
    for i, t in enumerate(sells):
        basis = basis_per_sell[i] if i < len(basis_per_sell) else t.price
        paired.append((t, basis))
    return paired


def compute_trade_stats(
    trades: list[TradeRecord],
    basis_per_sell: list[float],
) -> tuple[float, float, float, float]:
    """Return (win_rate, profit_factor, avg_win, avg_loss) from sell trades.

    Breakeven sells (`pnl == 0`) are counted as wins by convention.
    """
    paired = _pair_sells_with_basis(trades, basis_per_sell)
    if not paired:
        return 0.0, 0.0, 0.0, 0.0

    wins: list[float] = []
    losses: list[float] = []
    for t, basis in paired:
        pnl = (t.price - basis) * t.shares
        if pnl >= 0:
            wins.append(pnl)
        else:
            losses.append(pnl)

    total = len(wins) + len(losses)
    win_rate = len(wins) / total if total > 0 else 0.0
    gross_wins = sum(wins) if wins else 0.0
    gross_losses = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf") if gross_wins > 0 else 0.0
    avg_win = gross_wins / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0  # negative number
    return win_rate, profit_factor, avg_win, avg_loss


def compute_strategy_stats(
    trades: list[TradeRecord],
    basis_per_sell: list[float],
) -> list[StrategyStats]:
    """Compute per-strategy trade breakdown."""
    sell_basis: dict[int, float] = {id(t): b for t, b in _pair_sells_with_basis(trades, basis_per_sell)}

    by_strategy: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        by_strategy[t.strategy_name].append(t)

    stats: list[StrategyStats] = []
    for name, strades in sorted(by_strategy.items()):
        buys = [t for t in strades if t.direction == Direction.BUY]
        sells = [t for t in strades if t.direction == Direction.SELL]
        winning_sells = 0
        total_pnl = 0.0
        for t in sells:
            basis = sell_basis.get(id(t), t.price)
            pnl = (t.price - basis) * t.shares
            total_pnl += pnl
            if pnl >= 0:
                winning_sells += 1
        win_rate = winning_sells / len(sells) if sells else 0.0
        stats.append(
            StrategyStats(
                name=name,
                trades=len(strades),
                buys=len(buys),
                sells=len(sells),
                win_rate=round(win_rate, 4),
                pnl=round(total_pnl, 2),
            )
        )
    return stats


DEFAULT_TRAIN_PCT = 0.70


class BacktestEngine:
    def __init__(
        self,
        allocator: Allocator,
        rebalancer: Rebalancer,
        mechanical_strategies: list[Strategy] | None = None,
        constraints: AllocationConstraints | None = None,
        train_pct: float = DEFAULT_TRAIN_PCT,
        enable_split: bool = True,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self._allocator = allocator
        self._rebalancer = rebalancer
        self._mechanical = mechanical_strategies or []
        self._constraints = constraints or AllocationConstraints()
        self._train_pct = train_pct
        self._enable_split = enable_split
        self._log = log_fn or (lambda _msg: None)

    def run(
        self,
        portfolio: PortfolioConfig,
        price_data: dict[str, pd.Series],
        start: date,
        end: date,
    ) -> BacktestResult:
        trading_days = self._collect_trading_days(price_data, start, end)
        _, split_date = self._compute_split(trading_days)
        ticker_idx = self._build_ticker_index(price_data, start, end)
        state = self._init_positions(portfolio, price_data, trading_days, start, end)

        # Precompute strategy signals over full price arrays (one-time cost).
        full_prices = {ticker: idx.values for ticker, idx in ticker_idx.items()}
        self._allocator.precompute_signals(full_prices)

        deferred = self._find_deferred(portfolio, price_data, trading_days, start, end)

        self._simulate(
            state,
            portfolio,
            trading_days,
            ticker_idx,
            deferred,
            split_date,
        )

        return self._build_result(
            state,
            portfolio,
            price_data,
            trading_days,
            split_date,
        )

    # -- Phase helpers --------------------------------------------------------

    def _collect_trading_days(
        self,
        price_data: dict[str, pd.Series],
        start: date,
        end: date,
    ) -> list[date]:
        all_dates: set[date] = set()
        for series in price_data.values():
            all_dates.update(d for d in series.index if start <= d <= end)
        days = sorted(all_dates)
        if not days:
            msg = "No trading days found in date range"
            raise ValueError(msg)
        return days

    def _compute_split(
        self,
        trading_days: list[date],
    ) -> tuple[int, date | None]:
        if self._enable_split:
            split_idx = int(len(trading_days) * self._train_pct)
            if split_idx < len(trading_days):
                return split_idx, trading_days[split_idx]
        return len(trading_days), None

    def _build_ticker_index(
        self,
        price_data: dict[str, pd.Series],
        start: date,
        end: date,
    ) -> dict[str, _TickerIndex]:
        """Build per-ticker arrays keeping the warmup prefix intact.

        The returned array spans ``[max(first_available, start - warmup_bars), end]``
        so ``precompute_signals`` and the per-day pointer advance see a prefix
        of history before the user's ``start``. The simulation loop iterates
        over ``trading_days`` (which are already filtered to ``start..end``),
        so the pointer walks through warmup dates on day one of the sim.
        """
        warmup_bars = self._warmup_bars()
        index: dict[str, _TickerIndex] = {}
        for ticker, series in price_data.items():
            bounded = series[series.index <= end]
            if len(bounded) == 0:
                continue
            # Identify the first trading day inside the sim window.
            sim_mask = np.asarray(bounded.index >= start)
            if not sim_mask.any():
                continue
            sim_first_idx = int(np.argmax(sim_mask))
            # Keep up to ``warmup_bars`` prior bars for precompute.
            warmup_first_idx = max(0, sim_first_idx - warmup_bars)
            available_warmup = sim_first_idx - warmup_first_idx
            if warmup_bars > 0 and available_warmup < warmup_bars:
                self._log(
                    f"{ticker}: only {available_warmup} warmup bars available "
                    f"(requested {warmup_bars}) — strategies will score on "
                    f"partial history until enough bars accumulate"
                )
            sliced = bounded.iloc[warmup_first_idx:]
            index[ticker] = _TickerIndex(
                dates=list(sliced.index),
                values=np.asarray(sliced.values),
            )
        return index

    def _warmup_bars(self) -> int:
        """Max warmup required across allocator + mechanical strategies."""
        return max_warmup([*self._allocator.strategies, *self._mechanical])

    def _first_data_dates(
        self,
        price_data: dict[str, pd.Series],
        start: date,
    ) -> dict[str, date]:
        result: dict[str, date] = {}
        for ticker, series in price_data.items():
            in_range = series[series.index >= start]
            if len(in_range) > 0:
                result[ticker] = in_range.index[0]
        return result

    def _init_positions(
        self,
        portfolio: PortfolioConfig,
        price_data: dict[str, pd.Series],
        trading_days: list[date],
        start: date,
        end: date,
    ) -> _SimState:
        state = _SimState(cash=portfolio.available_cash)
        if portfolio.trading_restrictions:
            state.restriction_tracker = RestrictionTracker(
                portfolio.trading_restrictions,
            )
        first_dates = self._first_data_dates(price_data, start)

        for h in portfolio.holdings:
            if h.shares <= 0:
                continue

            if h.ticker not in first_dates:
                self._log(f"{h.ticker}: no price data in backtest range ({start} to {end}) — excluded")
                continue

            ticker_start = first_dates[h.ticker]
            if ticker_start <= trading_days[0]:
                entry_price = float(price_data[h.ticker][price_data[h.ticker].index >= start].iloc[0])
                state.positions[h.ticker] = h.shares
                state.bh_positions[h.ticker] = h.shares
                state.cost_basis[h.ticker] = entry_price
                state.purchase_dates[h.ticker] = [(h.shares, trading_days[0])]

        state.starting_value = state.cash + sum(
            shares * state.cost_basis[t]
            for t, shares in state.positions.items()
            if shares > 0 and t in state.cost_basis
        )
        state.twr_base_value = state.starting_value
        return state

    def _find_deferred(
        self,
        portfolio: PortfolioConfig,
        price_data: dict[str, pd.Series],
        trading_days: list[date],
        start: date,
        end: date,
    ) -> dict[str, float]:
        first_dates = self._first_data_dates(price_data, start)
        deferred: dict[str, float] = {}

        for h in portfolio.holdings:
            if h.shares <= 0 or h.ticker not in first_dates:
                continue
            ticker_start = first_dates[h.ticker]
            if ticker_start > trading_days[0]:
                deferred[h.ticker] = h.shares
                self._log(
                    f"{h.ticker}: data starts {ticker_start} "
                    f"(after backtest start {start}) — "
                    f"position deferred until first available date"
                )
        return deferred

    def _simulate(
        self,
        state: _SimState,
        portfolio: PortfolioConfig,
        trading_days: list[date],
        ticker_idx: dict[str, _TickerIndex],
        deferred: dict[str, float],
        split_date: date | None,
    ) -> None:
        for ticker, shares in deferred.items():
            state.bh_positions[ticker] = shares
            state.positions[ticker] = 0.0

        deferred_activated: set[str] = set()

        for day in trading_days:
            if state.last_day != day:
                state.last_day = day

            # Advance pointers and build current slices
            current_data: dict[str, np.ndarray] = {}
            for ticker, idx in ticker_idx.items():
                while idx.ptr < len(idx.dates) and idx.dates[idx.ptr] <= day:
                    idx.ptr += 1
                if idx.ptr > 0:
                    current_data[ticker] = idx.values[: idx.ptr]

            # Activate deferred holdings
            self._activate_deferred(
                state,
                deferred,
                deferred_activated,
                current_data,
                day,
            )

            # Capture split snapshot
            if split_date and day == split_date and state.split_value is None:
                split_val = state.cash + sum(
                    state.positions.get(t, 0) * float(current_data[t][-1]) for t in current_data
                )
                state.split_value = split_val
                state.split_bh_value = portfolio.available_cash + sum(
                    state.bh_positions.get(t, 0) * float(current_data[t][-1]) for t in current_data
                )
                # Close current TWR sub-period at the split boundary.
                if state.twr_base_value > 0:
                    state.twr_periods.append(split_val / state.twr_base_value)
                    state.twr_base_value = split_val
                state.twr_split_idx = len(state.twr_periods)

            # Phased allocator flow
            self._run_day(state, portfolio, current_data, day)

            # Record equity curve snapshot
            day_value = state.cash + sum(
                state.positions.get(t, 0) * float(current_data[t][-1])
                for t in current_data
                if state.positions.get(t, 0) > 0
            )
            state.equity_curve.append((day, day_value))

    def _activate_deferred(
        self,
        state: _SimState,
        deferred: dict[str, float],
        activated: set[str],
        current_data: dict[str, np.ndarray],
        day: date,
    ) -> None:
        for ticker, shares in deferred.items():
            if ticker in activated:
                continue
            if ticker in current_data:
                entry_price = float(current_data[ticker][-1])
                added_value = shares * entry_price

                # Treat deferred activation like a capital infusion for TWR:
                # close the current sub-period on existing positions (excluding
                # the newly activated ticker), then reset the base to include
                # the new position. Otherwise the new capital would be counted
                # as pure return by the closing final_value / twr_base ratio.
                pre_activation_value = state.cash + sum(
                    state.positions.get(t, 0) * float(current_data[t][-1])
                    for t in current_data
                    if state.positions.get(t, 0) > 0
                )
                if state.twr_base_value > 0:
                    state.twr_periods.append(pre_activation_value / state.twr_base_value)
                state.twr_base_value = pre_activation_value + added_value

                state.positions[ticker] = shares
                state.cost_basis[ticker] = entry_price
                state.purchase_dates[ticker] = [(shares, day)]
                state.starting_value += added_value
                activated.add(ticker)
                self._log(f"{ticker}: activated on {day} at ${entry_price:.2f} ({shares} shares)")

    def _run_day(
        self,
        state: _SimState,
        portfolio: PortfolioConfig,
        current_data: dict[str, np.ndarray],
        day: date,
    ) -> None:
        # Credit cash infusion if it lands on or before today.
        # For TWR: snapshot portfolio value before the infusion to close
        # the current sub-period, then reset the base after the infusion.
        infusion = portfolio.cash_infusion
        if infusion and infusion.next_date <= day:
            pre_infusion_value = state.cash + sum(
                state.positions.get(t, 0) * float(current_data[t][-1])
                for t in current_data
                if state.positions.get(t, 0) > 0
            )
            if state.twr_base_value > 0:
                state.twr_periods.append(pre_infusion_value / state.twr_base_value)
            state.cash += infusion.amount
            state.twr_base_value = pre_infusion_value + infusion.amount
            infusion.advance()

        # Build price arrays and current prices for active tickers
        active_tickers = [t for t in state.positions if state.positions.get(t, 0) > 0 or t in current_data]
        # Only include tickers that have price data
        active_tickers = [t for t in active_tickers if t in current_data]

        if not active_tickers:
            return

        current_prices: dict[str, float] = {}
        for ticker in active_tickers:
            current_prices[ticker] = float(current_data[ticker][-1])

        # Build per-ticker context (cost_basis for strategies that need it)
        context: dict[str, dict[str, Any]] = {}
        for ticker in active_tickers:
            ctx: dict[str, Any] = {}
            if ticker in state.cost_basis:
                ctx["cost_basis"] = state.cost_basis[ticker]
            context[ticker] = ctx

        # Phase 1-3: Allocator scores, blends, and applies vetoes
        allocation = self._allocator.allocate(
            active_tickers,
            current_data,
            context,
        )

        # Phase 4: Rebalancer diffs target vs current, generates orders
        positions = {t: state.positions.get(t, 0.0) for t in active_tickers}

        rebalance_orders = self._rebalancer.generate_orders(
            allocation,
            positions,
            current_prices,
            state.cash,
            self._constraints,
        )

        # Phase 5: Mechanical strategies generate intents
        mechanical_intents: list[MechanicalIntent] = []
        for strat in self._mechanical:
            for ticker in active_tickers:
                if ticker in current_data:
                    ticker_ctx = context.get(ticker, {})
                    intents = strat.generate_intents(
                        ticker,
                        current_data[ticker],
                        **ticker_ctx,
                    )
                    mechanical_intents.extend(intents)

        # Estimate post-rebalance cash for mechanical sizing
        sell_proceeds = sum(o.estimated_value for o in rebalance_orders if o.direction == Direction.SELL)
        buy_cost = sum(o.estimated_value for o in rebalance_orders if o.direction == Direction.BUY)
        post_rebalance_cash = state.cash + sell_proceeds - buy_cost

        mechanical_orders = self._rebalancer.size_mechanical(
            mechanical_intents,
            post_rebalance_cash,
            current_prices,
        )

        all_orders = rebalance_orders + mechanical_orders

        # Phase 6: Check trading restrictions, filter blocked
        filtered_orders: list[Order] = []
        for order in all_orders:
            if state.restriction_tracker and state.restriction_tracker.is_blocked(
                order.ticker,
                order.direction,
                day,
            ):
                continue
            filtered_orders.append(order)

        # Phase 7: Execute all orders
        for order in filtered_orders:
            if order.shares <= 0:
                continue
            # Snapshot cost basis before _execute / cash update; the position
            # may close inside _execute and have its basis popped below.
            sell_basis: float | None = None
            if order.direction == Direction.SELL:
                sell_basis = state.cost_basis.get(order.ticker, order.price)
            trade = self._execute(order, day, state)
            if trade is not None:
                state.trades.append(trade)
                if sell_basis is not None:
                    state.basis_per_sell.append(sell_basis)
                if state.restriction_tracker:
                    state.restriction_tracker.record_trade(
                        order.ticker,
                        order.direction,
                        day,
                    )
                if order.direction == Direction.BUY:
                    state.cash -= order.estimated_value
                    self._update_cost_basis(state, order)
                else:
                    state.cash += order.estimated_value
                    if state.positions.get(order.ticker, 0) == 0:
                        state.cost_basis.pop(order.ticker, None)

    def _update_cost_basis(self, state: _SimState, order: Order) -> None:
        tk = order.ticker
        old_shares = state.positions[tk] - order.shares
        old_cost = state.cost_basis.get(tk, 0.0)
        if state.positions[tk] > 0:
            state.cost_basis[tk] = (old_cost * old_shares + order.price * order.shares) / state.positions[tk]

    def _execute(
        self,
        order: Order,
        day: date,
        state: _SimState,
    ) -> TradeRecord | None:
        ticker = order.ticker
        strategy_name = order.context.source

        if order.direction == Direction.BUY:
            state.positions[ticker] = state.positions.get(ticker, 0) + order.shares
            state.purchase_dates.setdefault(ticker, []).append((order.shares, day))
            return TradeRecord(
                date=day,
                ticker=ticker,
                direction=Direction.BUY,
                shares=order.shares,
                price=order.price,
                strategy_name=strategy_name,
            )

        # SELL — FIFO for holding period
        holding_period = None
        lots = state.purchase_dates.get(ticker, [])
        if lots:
            days_held = (day - lots[0][1]).days
            holding_period = HoldingPeriod.LONG_TERM if days_held >= 365 else HoldingPeriod.SHORT_TERM
            remaining = order.shares
            while remaining > 0 and lots:
                lot_shares, lot_date = lots[0]
                if lot_shares <= remaining:
                    remaining -= lot_shares
                    lots.pop(0)
                else:
                    lots[0] = (lot_shares - remaining, lot_date)
                    remaining = 0

        state.positions[ticker] = max(
            0,
            state.positions.get(ticker, 0) - order.shares,
        )

        return TradeRecord(
            date=day,
            ticker=ticker,
            direction=Direction.SELL,
            shares=order.shares,
            price=order.price,
            strategy_name=strategy_name,
            holding_period=holding_period,
        )

    def _build_result(
        self,
        state: _SimState,
        portfolio: PortfolioConfig,
        price_data: dict[str, pd.Series],
        trading_days: list[date],
        split_date: date | None,
    ) -> BacktestResult:
        # Use prices at the last trading day within the backtest range, not the
        # last row of the raw series (which may extend beyond `end` when the
        # caller reuses one price_data dict across multiple sub-windows — e.g.
        # walk-forward fold evaluation).
        end_day = trading_days[-1]
        final_prices: dict[str, float] = {}
        for ticker, series in price_data.items():
            if len(series) == 0:
                continue
            in_range = series[series.index <= end_day]
            if len(in_range) > 0:
                final_prices[ticker] = float(in_range.iloc[-1])
        final_value = state.cash + sum(state.positions.get(t, 0) * p for t, p in final_prices.items())
        bh_value = portfolio.available_cash + sum(
            state.bh_positions.get(t, 0) * final_prices.get(t, 0) for t in state.bh_positions
        )

        # Close the final TWR sub-period and compound all periods.
        if state.twr_base_value > 0:
            state.twr_periods.append(final_value / state.twr_base_value)
        twr = 1.0
        for period_return in state.twr_periods:
            twr *= period_return
        twr -= 1.0

        sv = state.starting_value
        spbh = state.split_bh_value

        if split_date:
            train_trades = [t for t in state.trades if t.date < split_date]
            test_trades = [t for t in state.trades if t.date >= split_date]
        else:
            train_trades = list(state.trades)
            test_trades = []

        # Compute train/test TWR from sub-period boundaries.
        split_idx = state.twr_split_idx
        if split_idx is not None:
            train_twr = 1.0
            for p in state.twr_periods[:split_idx]:
                train_twr *= p
            train_twr -= 1.0
            test_twr = 1.0
            for p in state.twr_periods[split_idx:]:
                test_twr *= p
            test_twr -= 1.0
        else:
            train_twr = twr
            test_twr = twr

        train_return = train_twr
        test_return = test_twr
        train_bh_return = (spbh - sv) / sv if spbh is not None and sv > 0 else (bh_value - sv) / sv if sv > 0 else 0.0
        test_bh_return = (
            (bh_value - spbh) / spbh if spbh is not None and spbh > 0 else (bh_value - sv) / sv if sv > 0 else 0.0
        )

        # New metrics
        equity_curve = state.equity_curve
        total_days = (trading_days[-1] - trading_days[0]).days if len(trading_days) > 1 else 0
        cagr = compute_cagr(sv, final_value, total_days)
        max_drawdown = compute_max_drawdown(equity_curve)
        sharpe = compute_sharpe(equity_curve)
        sortino = compute_sortino(equity_curve)
        win_rate, profit_factor, avg_win, avg_loss = compute_trade_stats(state.trades, state.basis_per_sell)
        efficiency = test_return / train_return if split_date and train_return != 0 else 0.0
        strategy_stats = compute_strategy_stats(state.trades, state.basis_per_sell)

        return BacktestResult(
            trades=state.trades,
            final_value=round(final_value, 2),
            starting_value=round(sv, 2),
            buy_and_hold_value=round(bh_value, 2),
            train_trades=train_trades,
            test_trades=test_trades,
            train_return=round(train_return, 4),
            test_return=round(test_return, 4),
            train_bh_return=round(train_bh_return, 4),
            test_bh_return=round(test_bh_return, 4),
            split_date=split_date,
            twr=round(twr, 4),
            equity_curve=equity_curve,
            cagr=round(cagr, 4),
            max_drawdown=round(max_drawdown, 4),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 4) if not math.isinf(profit_factor) else float("inf"),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            efficiency_ratio=round(efficiency, 4),
            strategy_stats=strategy_stats,
        )


def write_backtest_csv(result: BacktestResult, path: Path) -> None:
    """Write backtest results to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["=== TRADE LOG ==="])
        writer.writerow(
            [
                "Date",
                "Ticker",
                "Direction",
                "Shares",
                "Price",
                "Strategy",
                "Holding Period",
            ]
        )
        for t in result.trades:
            writer.writerow(
                [
                    t.date.isoformat(),
                    t.ticker,
                    t.direction.value,
                    t.shares,
                    f"{t.price:.2f}",
                    t.strategy_name,
                    t.holding_period.value if t.holding_period else "",
                ]
            )

        writer.writerow([])
        writer.writerow(["=== SUMMARY ==="])
        writer.writerow(["Metric", "Value"])
        sv = result.starting_value
        total_return = (result.final_value - sv) / sv if sv > 0 else 0
        bh_return = (result.buy_and_hold_value - sv) / sv if sv > 0 else 0
        writer.writerow(["Starting Value", f"${sv:.2f}"])
        writer.writerow(["Final Value", f"${result.final_value:.2f}"])
        writer.writerow(["Total Return", f"{total_return:.2%}"])
        writer.writerow(["CAGR", f"{result.cagr:.2%}"])
        writer.writerow(["Time-Weighted Return", f"{result.twr:.2%}"])
        writer.writerow(["Buy & Hold Value", f"${result.buy_and_hold_value:.2f}"])
        writer.writerow(["Buy & Hold Return", f"{bh_return:.2%}"])
        writer.writerow(["Total Trades", len(result.trades)])

        writer.writerow([])
        writer.writerow(["=== RISK METRICS ==="])
        writer.writerow(["Max Drawdown", f"{result.max_drawdown:.2%}"])
        writer.writerow(["Sharpe Ratio", f"{result.sharpe_ratio:.4f}"])
        writer.writerow(["Sortino Ratio", f"{result.sortino_ratio:.4f}"])

        sells = [t for t in result.trades if t.direction == Direction.SELL]
        if sells:
            writer.writerow([])
            writer.writerow(["=== TRADE QUALITY ==="])
            writer.writerow(["Win Rate", f"{result.win_rate:.2%}"])
            pf = f"{result.profit_factor:.4f}" if not math.isinf(result.profit_factor) else "inf"
            writer.writerow(["Profit Factor", pf])
            writer.writerow(["Avg Win", f"${result.avg_win:.2f}"])
            writer.writerow(["Avg Loss", f"${result.avg_loss:.2f}"])

        if result.split_date:
            writer.writerow([])
            writer.writerow(["=== OUT-OF-SAMPLE SPLIT ==="])
            writer.writerow(["Split Date", result.split_date.isoformat()])
            writer.writerow(["Train Return", f"{result.train_return:.2%}"])
            writer.writerow(["Train B&H Return", f"{result.train_bh_return:.2%}"])
            writer.writerow(["Train Trades", len(result.train_trades)])
            writer.writerow(["Test Return", f"{result.test_return:.2%}"])
            writer.writerow(["Test B&H Return", f"{result.test_bh_return:.2%}"])
            writer.writerow(["Test Trades", len(result.test_trades)])
            writer.writerow(["Efficiency Ratio", f"{result.efficiency_ratio:.2%}"])

        if result.strategy_stats:
            writer.writerow([])
            writer.writerow(["=== STRATEGY BREAKDOWN ==="])
            writer.writerow(["Strategy", "Trades", "Buys", "Sells", "Win Rate", "P&L"])
            for s in result.strategy_stats:
                writer.writerow(
                    [
                        s.name,
                        s.trades,
                        s.buys,
                        s.sells,
                        f"{s.win_rate:.2%}" if s.sells > 0 else "",
                        f"${s.pnl:.2f}" if s.sells > 0 else "",
                    ]
                )
