"""Backtest engine — replays historical data through strategies."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from midas.allocator import AllocationResult, Allocator
from midas.models import (
    AllocationConstraints,
    Direction,
    HoldingPeriod,
    Order,
    PortfolioConfig,
    PositionLot,
    TradeRecord,
)
from midas.order_sizer import OrderSizer
from midas.restrictions import RestrictionTracker
from midas.strategies.base import ExitRule, max_warmup


@dataclass
class StrategyStats:
    """Per-strategy performance breakdown, optionally scoped to a ticker."""

    name: str
    ticker: str | None
    trades: int
    buys: int
    sells: int
    win_rate: float  # fraction of profitable sells
    pnl: float  # total realized P&L from sells


def aggregate_strategy_stats(stats: list[StrategyStats]) -> list[StrategyStats]:
    """Aggregate per-(strategy, ticker) stats into per-strategy totals."""
    by_strategy: dict[str, list[StrategyStats]] = defaultdict(list)
    for s in stats:
        by_strategy[s.name].append(s)
    result: list[StrategyStats] = []
    for name, group in sorted(by_strategy.items()):
        total_sells = sum(s.sells for s in group)
        total_pnl = sum(s.pnl for s in group)
        winning = sum(round(s.win_rate * s.sells) for s in group)
        agg_wr = winning / total_sells if total_sells > 0 else 0.0
        result.append(
            StrategyStats(
                name=name,
                ticker=None,
                trades=sum(s.trades for s in group),
                buys=sum(s.buys for s in group),
                sells=total_sells,
                win_rate=round(agg_wr, 4),
                pnl=round(total_pnl, 2),
            )
        )
    return result


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
    unrealized_pnl: float  # mark-to-market gain on positions still held at end
    unrealized_pnl_by_ticker: dict[str, float]  # per-ticker unrealized P&L
    basis_per_sell: list[float]  # cost basis for each SELL trade (parallel list)


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
    # Cost basis recorded at the moment each SELL trade was executed, in the
    # same order as SELL entries in `trades`. Each entry is the share-weighted
    # average of the FIFO lots actually consumed by that sell — not the
    # weighted-average of all lots — so partial sells of mixed-basis positions
    # attribute P&L to the correct lots. A parallel list (rather than a
    # (date, ticker)-keyed dict) is required so that multiple sells of the
    # same ticker on the same day — e.g. from different strategies — each get
    # their own basis snapshot instead of overwriting one another.
    basis_per_sell: list[float] = field(default_factory=list)
    bh_positions: dict[str, float] = field(default_factory=dict)
    # Per-ticker FIFO list of open lots. Used for cost-basis accounting
    # and FIFO holding-period classification on sells.
    lots: dict[str, list[PositionLot]] = field(default_factory=dict)
    # Aggregate per-ticker high-water mark for exit rule evaluation.
    # Updated each tick with the day's closing price.
    high_water_marks: dict[str, float] = field(default_factory=dict)
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


def _drawdown_series(equity_curve: list[tuple[date, float]]) -> list[float]:
    """Per-point drawdown (fraction from running peak) for each equity curve entry."""
    if not equity_curve:
        return []
    peak = equity_curve[0][1]
    result: list[float] = []
    for _, value in equity_curve:
        if value > peak:
            peak = value
        result.append((peak - value) / peak if peak > 0 else 0.0)
    return result


def compute_max_drawdown(equity_curve: list[tuple[date, float]]) -> float:
    """Maximum peak-to-trough percentage decline."""
    dd = _drawdown_series(equity_curve)
    return max(dd) if len(dd) >= 2 else 0.0


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
    """Compute per-(strategy, ticker) trade breakdown."""
    sell_basis: dict[int, float] = {id(t): b for t, b in _pair_sells_with_basis(trades, basis_per_sell)}

    by_key: dict[tuple[str, str], list[TradeRecord]] = defaultdict(list)
    for t in trades:
        by_key[(t.strategy_name, t.ticker)].append(t)

    stats: list[StrategyStats] = []
    for (name, ticker), strades in sorted(by_key.items()):
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
                ticker=ticker,
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
        order_sizer: OrderSizer,
        exit_rules: list[ExitRule] | None = None,
        constraints: AllocationConstraints | None = None,
        train_pct: float = DEFAULT_TRAIN_PCT,
        enable_split: bool = True,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self._allocator = allocator
        self._order_sizer = order_sizer
        self._exit_rules = exit_rules or []
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
        """Max warmup required across entry signals and exit rules."""
        return max_warmup([*self._allocator.strategies, *self._exit_rules])

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
                # Backtest seeds the cost basis from the start-day market
                # price, not the YAML ``cost_basis``. The YAML value is the
                # user's real purchase basis (used by the live engine and for
                # display) — using it here would let exit rules fire on
                # pre-backtest gains, distorting strategy performance.
                entry_price = float(price_data[h.ticker][price_data[h.ticker].index >= start].iloc[0])
                state.positions[h.ticker] = h.shares
                state.bh_positions[h.ticker] = h.shares
                state.lots[h.ticker] = [
                    PositionLot(
                        shares=h.shares,
                        purchase_date=trading_days[0],
                        cost_basis=entry_price,
                    )
                ]
                state.high_water_marks[h.ticker] = entry_price

        state.starting_value = state.cash + sum(
            lot.shares * lot.cost_basis for lots in state.lots.values() for lot in lots
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
                state.lots[ticker] = [
                    PositionLot(
                        shares=shares,
                        purchase_date=day,
                        cost_basis=entry_price,
                    )
                ]
                state.high_water_marks[ticker] = entry_price
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

        # Compute current portfolio weights so the allocator can hold
        # (not drift-correct) tickers whose entry signals don't score today.
        # Pass None (not {}) when the denominator is zero so the allocator
        # falls back to its equal-weight baseline rather than anchoring held
        # tickers at 0.
        total_value = state.cash + sum(state.positions.get(t, 0.0) * current_prices[t] for t in active_tickers)
        current_weights: dict[str, float] | None = None
        if total_value > 0:
            current_weights = {
                t: (state.positions.get(t, 0.0) * current_prices[t]) / total_value for t in active_tickers
            }

        # Update aggregate per-ticker high-water marks before exit rules
        # evaluate. TrailingStop reads HWM to size its drawdown threshold.
        for ticker in active_tickers:
            if state.positions.get(ticker, 0.0) <= 0:
                continue
            px = current_prices[ticker]
            prev = state.high_water_marks.get(ticker, 0.0)
            if px > prev:
                state.high_water_marks[ticker] = px

        # Phase 1: Allocator scores entry signals and blends to target weights.
        allocation = self._allocator.allocate(
            active_tickers,
            current_data,
            current_weights=current_weights,
        )

        positions = {t: state.positions.get(t, 0.0) for t in active_tickers}

        # Phase 2: Exit rules clamp proposed targets downward (LEAN pattern).
        # Each rule can only reduce a target, never increase. First clamper
        # wins attribution for that ticker.
        clamped_targets = dict(allocation.targets)
        clamp_attribution: dict[str, tuple[str, str]] = {}
        for rule in self._exit_rules:
            for ticker in active_tickers:
                if state.positions.get(ticker, 0.0) <= 0:
                    continue
                proposed = clamped_targets.get(ticker, 0.0)
                if proposed <= 0:
                    continue
                cost_basis = self._aggregate_cost_basis(state.lots.get(ticker, []))
                hwm = state.high_water_marks.get(ticker, 0.0)
                clamped = rule.clamp_target(ticker, proposed, current_data[ticker], cost_basis, hwm)
                if clamped < proposed:
                    clamped_targets[ticker] = clamped
                    if ticker not in clamp_attribution:
                        reason = rule.clamp_reason(ticker, current_data[ticker], cost_basis, hwm)
                        clamp_attribution[ticker] = (rule.name, reason)

        # Phase 3: Size sells from clamped targets and filter restriction-blocked
        # sells *before* computing post-sell cash. Otherwise a blocked sell would
        # leak phantom proceeds into the buy pass and the cash balance could
        # go negative when the buy fills but the sell didn't.
        exit_orders = self._order_sizer.size_sells(
            clamped_targets,
            positions,
            current_prices,
            total_value,
            clamp_attribution,
        )
        if state.restriction_tracker:
            exit_orders = [
                o for o in exit_orders if not state.restriction_tracker.is_blocked(o.ticker, o.direction, day)
            ]
        sell_proceeds = sum(o.estimated_value for o in exit_orders)
        post_sell_cash = state.cash + sell_proceeds

        # Build clamped allocation for buy sizing so that buy-side doesn't
        # try to buy tickers that were just clamped to 0.
        clamped_allocation = AllocationResult(
            targets=clamped_targets,
            contributions=allocation.contributions,
            blended_scores=allocation.blended_scores,
        )

        # Phase 4: Size buys against post-sell cash, then filter restrictions.
        # ``total_value`` is the same denominator the allocator used for
        # ``current_weights``; passing it through keeps the per-ticker delta
        # math consistent so held tickers don't fire phantom buys when sells
        # earlier in the tick freed cash and shifted post-sell weights.
        buy_orders = self._order_sizer.size_buys(
            clamped_allocation,
            positions,
            current_prices,
            post_sell_cash,
            self._constraints,
            total_value=total_value,
        )
        if state.restriction_tracker:
            buy_orders = [o for o in buy_orders if not state.restriction_tracker.is_blocked(o.ticker, o.direction, day)]

        # Phase 5: Execute sells first (so proceeds land in cash before buys),
        # then buys. ``_execute`` emits one TradeRecord per holding-period
        # group on sells, so mixed-lot sells crossing the 365-day boundary
        # split into separate ST and LT records with per-group cost basis.
        for order in exit_orders:
            if order.shares <= 0:
                continue
            records = self._execute(order, day, state)
            if not records:
                continue
            for trade, basis in records:
                state.trades.append(trade)
                state.basis_per_sell.append(basis)
            if state.restriction_tracker:
                state.restriction_tracker.record_trade(order.ticker, order.direction, day)
            state.cash += order.estimated_value

        for order in buy_orders:
            if order.shares <= 0:
                continue
            records = self._execute(order, day, state)
            if not records:
                continue
            state.trades.append(records[0][0])
            if state.restriction_tracker:
                state.restriction_tracker.record_trade(order.ticker, order.direction, day)
            state.cash -= order.estimated_value

    @staticmethod
    def _aggregate_cost_basis(lots: list[PositionLot]) -> float:
        """Share-weighted average cost basis across all open lots."""
        total_shares = sum(lot.shares for lot in lots)
        if total_shares <= 0:
            return 0.0
        return sum(lot.shares * lot.cost_basis for lot in lots) / total_shares

    @staticmethod
    def _fifo_consumed_basis(lots: list[PositionLot], shares: float) -> float:
        """Share-weighted cost basis of the first *shares* shares in FIFO order.

        Mirrors what ``_execute`` will pop off the lot list, without mutating
        it. Used to record the actual basis for the lots leaving the position
        on a sell, rather than the weighted average of all open lots.
        """
        if shares <= 0 or not lots:
            return 0.0
        remaining = shares
        weighted = 0.0
        consumed = 0.0
        for lot in lots:
            if remaining <= 0:
                break
            take = min(lot.shares, remaining)
            weighted += take * lot.cost_basis
            consumed += take
            remaining -= take
        return weighted / consumed if consumed > 0 else 0.0

    def _execute(
        self,
        order: Order,
        day: date,
        state: _SimState,
    ) -> list[tuple[TradeRecord, float]]:
        """Execute an order and return ``(TradeRecord, cost_basis)`` pairs.

        Buys return a single pair with ``cost_basis=0.0`` (unused). Sells
        FIFO-consume the lot list and emit one pair per holding-period
        bucket, so a sell straddling the 365-day ST/LT boundary produces
        two records — one short-term, one long-term — each with its own
        share count and weighted cost basis.
        """
        ticker = order.ticker
        strategy_name = order.context.source

        if order.direction == Direction.BUY:
            state.positions[ticker] = state.positions.get(ticker, 0) + order.shares
            state.lots.setdefault(ticker, []).append(
                PositionLot(shares=order.shares, purchase_date=day, cost_basis=order.price)
            )
            return [
                (
                    TradeRecord(
                        date=day,
                        ticker=ticker,
                        direction=Direction.BUY,
                        shares=order.shares,
                        price=order.price,
                        strategy_name=strategy_name,
                    ),
                    0.0,
                )
            ]

        # SELL — FIFO consume lots, bucketing each consumed slice into the
        # short-term or long-term group based on its own purchase date.
        lots = state.lots.get(ticker, [])
        if not lots:
            return []

        st_shares = 0.0
        st_weighted_basis = 0.0
        lt_shares = 0.0
        lt_weighted_basis = 0.0

        remaining = order.shares
        while remaining > 0 and lots:
            lot = lots[0]
            take = min(lot.shares, remaining)
            if lot.purchase_date is not None and (day - lot.purchase_date).days >= 365:
                lt_shares += take
                lt_weighted_basis += take * lot.cost_basis
            else:
                st_shares += take
                st_weighted_basis += take * lot.cost_basis

            if lot.shares <= remaining:
                remaining -= lot.shares
                lots.pop(0)
            else:
                lots[0] = PositionLot(
                    shares=lot.shares - remaining,
                    purchase_date=lot.purchase_date,
                    cost_basis=lot.cost_basis,
                )
                remaining = 0

        new_position = state.positions.get(ticker, 0) - order.shares
        # ``size_sells`` caps at held shares, so reaching _execute with a
        # sell larger than the position is a logic bug — not a float rounding
        # blip — and would fabricate cash if silently clamped to 0. Fail loud.
        assert new_position >= 0, (
            f"sell exceeds position on {ticker}: {order.shares} shares against {state.positions.get(ticker, 0)} held"
        )
        state.positions[ticker] = new_position

        # Full exit resets the high-water mark. Otherwise a subsequent
        # re-entry would inherit the old peak and TrailingStop (or any
        # future HWM-driven rule) could misfire on day 1 of the new
        # position against a price that the new lot has never seen.
        if new_position == 0:
            state.high_water_marks.pop(ticker, None)

        records: list[tuple[TradeRecord, float]] = []
        if st_shares > 0:
            records.append(
                (
                    TradeRecord(
                        date=day,
                        ticker=ticker,
                        direction=Direction.SELL,
                        shares=st_shares,
                        price=order.price,
                        strategy_name=strategy_name,
                        holding_period=HoldingPeriod.SHORT_TERM,
                    ),
                    st_weighted_basis / st_shares,
                )
            )
        if lt_shares > 0:
            records.append(
                (
                    TradeRecord(
                        date=day,
                        ticker=ticker,
                        direction=Direction.SELL,
                        shares=lt_shares,
                        price=order.price,
                        strategy_name=strategy_name,
                        holding_period=HoldingPeriod.LONG_TERM,
                    ),
                    lt_weighted_basis / lt_shares,
                )
            )
        return records

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

        # Unrealized P&L: mark-to-market gain on positions still held at end.
        unrealized_pnl_by_ticker: dict[str, float] = {}
        for ticker, lots in state.lots.items():
            ticker_pnl = sum(lot.shares * (final_prices.get(ticker, 0.0) - lot.cost_basis) for lot in lots)
            if lots:
                unrealized_pnl_by_ticker[ticker] = round(ticker_pnl, 2)
        unrealized_pnl = sum(unrealized_pnl_by_ticker.values())

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
            unrealized_pnl=round(unrealized_pnl, 2),
            unrealized_pnl_by_ticker=unrealized_pnl_by_ticker,
            basis_per_sell=state.basis_per_sell,
        )


def write_backtest_results(result: BacktestResult, output_dir: Path) -> None:
    """Write backtest results to a directory of machine-readable files."""
    if output_dir.is_file():
        msg = f"Output path '{output_dir}' is an existing file, not a directory"
        raise FileExistsError(msg)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_trades_csv(result, output_dir / "trades.csv")
    _write_equity_curve_csv(result, output_dir / "equity_curve.csv")
    _write_summary_json(result, output_dir / "summary.json")
    _write_strategy_breakdown_csv(result, output_dir / "strategy_breakdown.csv")


def _write_trades_csv(result: BacktestResult, path: Path) -> None:
    sell_basis = {id(t): b for t, b in _pair_sells_with_basis(result.trades, result.basis_per_sell)}
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "date",
                "ticker",
                "direction",
                "shares",
                "price",
                "strategy",
                "holding_period",
                "cost_basis",
                "realized_pnl",
                "return_pct",
            ]
        )
        for t in result.trades:
            common = [
                t.date.isoformat(),
                t.ticker,
                t.direction.value,
                t.shares,
                t.price,
                t.strategy_name,
                t.holding_period.value if t.holding_period else "",
            ]
            if t.direction == Direction.SELL:
                basis = sell_basis.get(id(t), t.price)
                pnl = round((t.price - basis) * t.shares, 4)
                ret = round((t.price - basis) / basis, 6) if basis != 0 else 0.0
                writer.writerow([*common, round(basis, 4), pnl, ret])
            else:
                writer.writerow([*common, "", "", ""])


def _write_equity_curve_csv(result: BacktestResult, path: Path) -> None:
    drawdowns = _drawdown_series(result.equity_curve)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "nav", "drawdown"])
        for (d, nav), dd in zip(result.equity_curve, drawdowns, strict=True):
            writer.writerow([d.isoformat(), round(nav, 2), round(dd, 6)])


def _write_summary_json(result: BacktestResult, path: Path) -> None:
    sv = result.starting_value
    total_return = (result.final_value - sv) / sv if sv > 0 else 0.0
    bh_return = (result.buy_and_hold_value - sv) / sv if sv > 0 else 0.0

    summary: dict[str, object] = {
        "starting_value": sv,
        "final_value": result.final_value,
        "total_return": round(total_return, 6),
        "cagr": result.cagr,
        "twr": result.twr,
        "buy_and_hold_value": result.buy_and_hold_value,
        "buy_and_hold_return": round(bh_return, 6),
        "total_trades": len(result.trades),
        "max_drawdown": result.max_drawdown,
        "sharpe_ratio": result.sharpe_ratio,
        "sortino_ratio": result.sortino_ratio,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor if not math.isinf(result.profit_factor) else "inf",
        "avg_win": result.avg_win,
        "avg_loss": result.avg_loss,
        "unrealized_pnl": result.unrealized_pnl,
        "efficiency_ratio": result.efficiency_ratio,
    }

    if result.split_date:
        summary["split"] = {
            "date": result.split_date.isoformat(),
            "train_return": result.train_return,
            "test_return": result.test_return,
            "train_bh_return": result.train_bh_return,
            "test_bh_return": result.test_bh_return,
            "train_trades": len(result.train_trades),
            "test_trades": len(result.test_trades),
        }

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def _write_strategy_breakdown_csv(result: BacktestResult, path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "ticker", "trades", "buys", "sells", "win_rate", "pnl"])

        # Per-(strategy, ticker) rows
        for s in result.strategy_stats:
            writer.writerow(
                [
                    s.name,
                    s.ticker,
                    s.trades,
                    s.buys,
                    s.sells,
                    round(s.win_rate, 4) if s.sells > 0 else "",
                    round(s.pnl, 2) if s.sells > 0 else "",
                ]
            )

        # Aggregate per-strategy rows
        for a in aggregate_strategy_stats(result.strategy_stats):
            writer.writerow(
                [
                    a.name,
                    "*",
                    a.trades,
                    a.buys,
                    a.sells,
                    a.win_rate if a.sells > 0 else "",
                    a.pnl if a.sells > 0 else "",
                ]
            )

        # Per-ticker open positions
        for ticker, pnl in sorted(result.unrealized_pnl_by_ticker.items()):
            writer.writerow(["Open Positions (Unrealized)", ticker, "", "", "", "", round(pnl, 2)])
        writer.writerow(["Open Positions (Unrealized)", "*", "", "", "", "", round(result.unrealized_pnl, 2)])
