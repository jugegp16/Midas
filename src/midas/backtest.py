"""Backtest engine — replays historical data through strategies."""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from midas.agent import Agent
from midas.models import (
    Direction,
    HoldingPeriod,
    Order,
    PortfolioConfig,
    TradingRestrictions,
    TradeRecord,
)
from midas.restrictions import RestrictionTracker
from midas.sizing import SizingEngine


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
    bh_positions: dict[str, float] = field(default_factory=dict)
    purchase_dates: dict[str, list[tuple[float, date]]] = field(default_factory=dict)
    cash: float = 0.0
    starting_value: float = 0.0
    trades: list[TradeRecord] = field(default_factory=list)
    daily_deployed: float = 0.0
    last_day: date | None = None
    split_value: float | None = None
    split_bh_value: float | None = None
    restriction_tracker: RestrictionTracker | None = None


class BacktestEngine:
    def __init__(
        self,
        agents: list[Agent],
        sizing_engine: SizingEngine | None = None,
        train_pct: float = 0.70,
        enable_split: bool = True,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self._agents = agents
        self._sizing = sizing_engine or SizingEngine()
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

        # Deferred holdings: data starts after backtest start
        deferred = self._find_deferred(portfolio, price_data, trading_days, start, end)

        self._simulate(
            state, portfolio, trading_days, ticker_idx, deferred, split_date,
        )

        return self._build_result(
            state, portfolio, price_data, trading_days, split_date,
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
        self, trading_days: list[date],
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
        index: dict[str, _TickerIndex] = {}
        for ticker, series in price_data.items():
            in_range = series[
                (series.index >= start) & (series.index <= end)  # type: ignore[operator]
            ]
            if len(in_range) > 0:
                index[ticker] = _TickerIndex(
                    dates=list(in_range.index),
                    values=in_range.values,
                )
        return index

    def _first_data_dates(
        self,
        price_data: dict[str, pd.Series],
        start: date,
    ) -> dict[str, date]:
        result: dict[str, date] = {}
        for ticker, series in price_data.items():
            in_range = series[series.index >= start]  # type: ignore[operator]
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
                self._log(
                    f"{h.ticker}: no price data in backtest range "
                    f"({start} to {end}) — excluded"
                )
                continue

            ticker_start = first_dates[h.ticker]
            if ticker_start <= trading_days[0]:
                entry_price = float(
                    price_data[h.ticker][
                        price_data[h.ticker].index >= start  # type: ignore[operator]
                    ].iloc[0]
                )
                state.positions[h.ticker] = h.shares
                state.bh_positions[h.ticker] = h.shares
                state.cost_basis[h.ticker] = entry_price
                state.purchase_dates[h.ticker] = [(h.shares, trading_days[0])]

        state.starting_value = state.cash + sum(
            shares * state.cost_basis[t]
            for t, shares in state.positions.items()
            if shares > 0 and t in state.cost_basis
        )
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
                # Track in state via caller — but set up bh baseline now
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
        # Set up deferred bh positions
        for ticker, shares in deferred.items():
            state.bh_positions[ticker] = shares
            state.positions[ticker] = 0.0

        deferred_activated: set[str] = set()

        for day in trading_days:
            if state.last_day != day:
                state.daily_deployed = 0.0
                state.last_day = day

            # Advance pointers and build current slices
            current_data: dict[str, np.ndarray] = {}
            for ticker, idx in ticker_idx.items():
                while idx.ptr < len(idx.dates) and idx.dates[idx.ptr] <= day:
                    idx.ptr += 1
                if idx.ptr > 0:
                    current_data[ticker] = idx.values[:idx.ptr]

            # Activate deferred holdings
            self._activate_deferred(
                state, deferred, deferred_activated, current_data, day,
            )

            # Capture split snapshot
            if split_date and day == split_date and state.split_value is None:
                state.split_value = state.cash + sum(
                    state.positions.get(t, 0) * float(current_data[t][-1])
                    for t in current_data
                )
                state.split_bh_value = portfolio.available_cash + sum(
                    state.bh_positions.get(t, 0) * float(current_data[t][-1])
                    for t in current_data
                )

            # Run agents and execute orders
            self._run_day(state, portfolio, current_data, day)

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
                state.positions[ticker] = shares
                state.cost_basis[ticker] = entry_price
                state.purchase_dates[ticker] = [(shares, day)]
                state.starting_value += shares * entry_price
                activated.add(ticker)
                self._log(
                    f"{ticker}: activated on {day} at "
                    f"${entry_price:.2f} ({shares} shares)"
                )

    def _run_day(
        self,
        state: _SimState,
        portfolio: PortfolioConfig,
        current_data: dict[str, np.ndarray],
        day: date,
    ) -> None:
        for agent in self._agents:
            signals = agent.run(
                portfolio, current_data,
                today=day,
                cost_basis_overrides=state.cost_basis,
            )
            for signal in signals:
                order = self._sizing.size_order(
                    signal=signal,
                    available_cash=state.cash,
                    position_shares=state.positions.get(signal.ticker, 0),
                    cash_infusion=portfolio.cash_infusion,
                    daily_deployed=state.daily_deployed,
                    today=day,
                )
                if order is None or order.shares == 0:
                    continue

                if state.restriction_tracker and state.restriction_tracker.is_blocked(
                    order.ticker, order.direction, day,
                ):
                    continue

                trade = self._execute(order, day, state)
                if trade is not None:
                    state.trades.append(trade)
                    if state.restriction_tracker:
                        state.restriction_tracker.record_trade(
                            order.ticker, order.direction, day,
                        )
                    if order.direction == Direction.BUY:
                        state.cash -= order.estimated_value
                        state.daily_deployed += order.estimated_value
                        self._update_cost_basis(state, order)
                    else:
                        state.cash += order.estimated_value
                        if state.positions.get(order.ticker, 0) == 0:
                            state.cost_basis.pop(order.ticker, None)

    def _update_cost_basis(self, state: _SimState, order: Order) -> None:
        tk = order.ticker
        old_shares = state.positions[tk] - order.shares
        old_cost = state.cost_basis.get(tk, 0.0)
        state.cost_basis[tk] = (
            old_cost * old_shares + order.signal.price * order.shares
        ) / state.positions[tk]

    def _execute(
        self,
        order: Order,
        day: date,
        state: _SimState,
    ) -> TradeRecord | None:
        ticker = order.ticker

        if order.direction == Direction.BUY:
            state.positions[ticker] = state.positions.get(ticker, 0) + order.shares
            state.purchase_dates.setdefault(ticker, []).append((order.shares, day))
            return TradeRecord(
                date=day,
                ticker=ticker,
                direction=Direction.BUY,
                shares=order.shares,
                price=order.signal.price,
                strategy_name=order.signal.strategy_name,
            )

        # SELL — FIFO for holding period
        holding_period = None
        lots = state.purchase_dates.get(ticker, [])
        if lots:
            days_held = (day - lots[0][1]).days
            holding_period = (
                HoldingPeriod.LONG_TERM
                if days_held >= 365
                else HoldingPeriod.SHORT_TERM
            )
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
            0, state.positions.get(ticker, 0) - order.shares,
        )

        return TradeRecord(
            date=day,
            ticker=ticker,
            direction=Direction.SELL,
            shares=order.shares,
            price=order.signal.price,
            strategy_name=order.signal.strategy_name,
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
        final_prices = {
            t: float(s.iloc[-1])
            for t, s in price_data.items()
            if len(s) > 0
        }
        final_value = state.cash + sum(
            state.positions.get(t, 0) * p for t, p in final_prices.items()
        )
        bh_value = portfolio.available_cash + sum(
            state.bh_positions.get(t, 0) * final_prices.get(t, 0)
            for t in state.bh_positions
        )

        sv = state.starting_value
        spv = state.split_value
        spbh = state.split_bh_value

        train_trades = [t for t in state.trades if split_date and t.date < split_date]
        test_trades = [t for t in state.trades if split_date and t.date >= split_date]

        train_return = (spv - sv) / sv if spv is not None and sv > 0 else 0.0
        test_return = (
            (final_value - spv) / spv
            if spv is not None and spv > 0
            else (final_value - sv) / sv if sv > 0 else 0.0
        )
        train_bh_return = (spbh - sv) / sv if spbh is not None and sv > 0 else 0.0
        test_bh_return = (
            (bh_value - spbh) / spbh
            if spbh is not None and spbh > 0
            else (bh_value - sv) / sv if sv > 0 else 0.0
        )

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
        )


def write_backtest_csv(result: BacktestResult, path: Path) -> None:
    """Write backtest results to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["=== TRADE LOG ==="])
        writer.writerow([
            "Date", "Ticker", "Direction", "Shares", "Price",
            "Strategy", "Holding Period",
        ])
        for t in result.trades:
            writer.writerow([
                t.date.isoformat(),
                t.ticker,
                t.direction.value,
                t.shares,
                f"{t.price:.2f}",
                t.strategy_name,
                t.holding_period.value if t.holding_period else "",
            ])

        writer.writerow([])
        writer.writerow(["=== SUMMARY ==="])
        writer.writerow(["Metric", "Value"])
        sv = result.starting_value
        total_return = (result.final_value - sv) / sv if sv > 0 else 0
        bh_return = (result.buy_and_hold_value - sv) / sv if sv > 0 else 0
        writer.writerow(["Starting Value", f"${sv:.2f}"])
        writer.writerow(["Final Value", f"${result.final_value:.2f}"])
        writer.writerow(["Total Return", f"{total_return:.2%}"])
        writer.writerow(["Buy & Hold Value", f"${result.buy_and_hold_value:.2f}"])
        writer.writerow(["Buy & Hold Return", f"{bh_return:.2%}"])
        writer.writerow(["Total Trades", len(result.trades)])

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
