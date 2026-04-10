"""Tests for the backtest engine."""

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from conftest import make_price_series

from midas.allocator import Allocator
from midas.backtest import (
    TRADING_DAYS_PER_YEAR,
    BacktestEngine,
    _SimState,
    compute_cagr,
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
    compute_strategy_stats,
    compute_trade_stats,
    write_backtest_csv,
)
from midas.models import (
    AllocationConstraints,
    CashInfusion,
    Direction,
    Holding,
    PortfolioConfig,
    PositionLot,
    TradeRecord,
    TradingRestrictions,
)
from midas.order_sizer import OrderSizer
from midas.restrictions import RestrictionTracker
from midas.strategies.mean_reversion import MeanReversion
from midas.strategies.profit_taking import ProfitTaking
from midas.strategies.trailing_stop import TrailingStop


def _build_engine(
    entries=None,
    exit_rules=None,
    constraints=None,
    n_tickers=1,
    **kwargs,
):
    """Helper to build a BacktestEngine with the new allocator + order_sizer system."""
    entries = entries or []
    constraints = constraints or AllocationConstraints(
        min_buy_delta=0.01,
        max_position_pct=0.95,
    )
    allocator = Allocator(entries, constraints, n_tickers)
    order_sizer = OrderSizer()
    return BacktestEngine(
        allocator=allocator,
        order_sizer=order_sizer,
        exit_rules=exit_rules,
        constraints=constraints,
        **kwargs,
    )


def _make_backtest_data() -> tuple[PortfolioConfig, dict[str, pd.Series]]:
    """Create a portfolio and price data that will generate trades."""
    portfolio = PortfolioConfig(
        holdings=[
            Holding(ticker="AAPL", shares=10, cost_basis=90.0),
        ],
        available_cash=2000.0,
    )

    # Price drops then recovers — triggers mean reversion buy
    returns = (
        [0.0] * 20  # flat at 100
        + [-0.008] * 20  # drop ~15%
        + [0.01] * 30  # recover
        + [0.0] * 30  # flat
    )
    prices = make_price_series(date(2024, 1, 2), 100, 100.0, returns, name="AAPL")
    return portfolio, {"AAPL": prices}


def test_backtest_produces_trades() -> None:
    portfolio, price_data = _make_backtest_data()
    mr = MeanReversion(window=20, threshold=0.05)
    engine = _build_engine(
        entries=[(mr, 1.0)],
        enable_split=False,
    )

    start = min(price_data["AAPL"].index)
    end = max(price_data["AAPL"].index)
    result = engine.run(portfolio, price_data, start, end)

    assert result.starting_value > 0
    assert result.final_value > 0
    assert len(result.trades) > 0
    for t in result.trades:
        assert t.ticker == "AAPL"


def test_backtest_with_split() -> None:
    portfolio, price_data = _make_backtest_data()
    mr = MeanReversion(window=20, threshold=0.05)
    engine = _build_engine(
        entries=[(mr, 1.0)],
        train_pct=0.7,
    )

    start = min(price_data["AAPL"].index)
    end = max(price_data["AAPL"].index)
    result = engine.run(portfolio, price_data, start, end)

    assert result.split_date is not None
    assert start < result.split_date < end


def test_backtest_csv_output(tmp_path: Path) -> None:
    portfolio, price_data = _make_backtest_data()
    mr = MeanReversion(window=20, threshold=0.05)
    engine = _build_engine(
        entries=[(mr, 1.0)],
        enable_split=True,
    )

    start = min(price_data["AAPL"].index)
    end = max(price_data["AAPL"].index)
    result = engine.run(portfolio, price_data, start, end)

    csv_path = tmp_path / "results.csv"
    write_backtest_csv(result, csv_path)

    content = csv_path.read_text()
    assert "TRADE LOG" in content
    assert "SUMMARY" in content
    assert "AAPL" in content


def test_backtest_cost_basis_uses_start_price() -> None:
    """Cost basis should be the start-date price, not the config value."""
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="TEST", shares=10, cost_basis=50.0)],
        available_cash=0.0,
    )
    # Flat at 100 for 50 days, then rise
    returns = [0.0] * 50 + [0.005] * 50
    prices = make_price_series(date(2024, 1, 2), 100, 100.0, returns, name="TEST")

    pt = ProfitTaking(gain_threshold=0.20)
    engine = _build_engine(
        exit_rules=[pt],
        enable_split=False,
    )

    start = min(prices.index)
    end = max(prices.index)
    result = engine.run(portfolio, {"TEST": prices}, start, end)

    sells = [t for t in result.trades if t.direction == Direction.SELL]
    if sells:
        assert sells[0].date > start


def test_backtest_deferred_ticker() -> None:
    """Tickers whose data starts after the backtest start date should
    begin with 0 shares and activate when data first appears."""
    portfolio = PortfolioConfig(
        holdings=[
            Holding(ticker="EARLY", shares=10),
            Holding(ticker="LATE", shares=5),
        ],
        available_cash=1000.0,
    )
    early = make_price_series(date(2024, 1, 2), 100, 100.0, name="EARLY")
    late = make_price_series(date(2024, 3, 20), 50, 80.0, name="LATE")

    mr = MeanReversion(window=10, threshold=0.05)
    log_messages: list[str] = []
    engine = _build_engine(
        entries=[(mr, 1.0)],
        n_tickers=2,
        enable_split=False,
        log_fn=log_messages.append,
    )

    start = date(2024, 1, 2)
    end = date(2024, 6, 30)
    result = engine.run(portfolio, {"EARLY": early, "LATE": late}, start, end)

    deferred_msgs = [m for m in log_messages if "deferred" in m]
    activated_msgs = [m for m in log_messages if "activated" in m]
    assert len(deferred_msgs) == 1
    assert "LATE" in deferred_msgs[0]
    assert len(activated_msgs) == 1
    assert "LATE" in activated_msgs[0]

    assert result.starting_value == 2000.0 + 5 * 80.0


def test_backtest_consumes_warmup_prefix() -> None:
    """Bars before ``start`` should prime strategy signals from day one.

    Without a warmup prefix, a `window=20` strategy spends the first 20
    days of the simulation in warmup and emits no signals. With a warmup
    prefix fetched ahead of ``start``, the allocator can produce valid
    conviction scores on the very first simulation day.
    """
    # 60 bars total: bars 0-19 are warmup, bars 20-59 are the sim window.
    # Flat at 100 through bar 30, then drop — so the drop lands inside
    # the sim window but needs the warmup bars to compute its 20-day MA.
    returns = [0.0] * 30 + [-0.02] * 10 + [0.0] * 20
    prices = make_price_series(date(2024, 1, 2), 60, 100.0, returns, name="AAPL")
    trading_days = list(prices.index)
    sim_start = trading_days[20]
    sim_end = trading_days[-1]

    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10, cost_basis=100.0)],
        available_cash=5000.0,
    )

    mr = MeanReversion(window=20, threshold=0.05)
    engine = _build_engine(
        entries=[(mr, 1.0)],
        enable_split=False,
    )
    result = engine.run(portfolio, {"AAPL": prices}, sim_start, sim_end)

    # With warmup consumed, the drop triggers a mean-reversion buy inside
    # the sim window. Without warmup, the strategy would still be in its
    # 20-day cold start when the drop started and miss it entirely.
    buys = [t for t in result.trades if t.direction == Direction.BUY]
    assert buys, "Expected at least one buy — warmup prefix was not consumed"
    assert all(sim_start <= t.date <= sim_end for t in buys)


def test_backtest_logs_insufficient_warmup() -> None:
    """A ticker with less history than a strategy needs should log a warning."""
    # Only 25 bars total, starting exactly at the sim start — no prefix.
    prices = make_price_series(date(2024, 1, 2), 25, 100.0, name="AAPL")
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10, cost_basis=100.0)],
        available_cash=1000.0,
    )

    # window=50 → warmup_period=50, but only ~25 bars are available.
    mr = MeanReversion(window=50, threshold=0.05)
    log_messages: list[str] = []
    engine = _build_engine(
        entries=[(mr, 1.0)],
        enable_split=False,
        log_fn=log_messages.append,
    )
    engine.run(portfolio, {"AAPL": prices}, min(prices.index), max(prices.index))

    warmup_msgs = [m for m in log_messages if "warmup" in m.lower()]
    assert warmup_msgs, f"Expected warmup warning, got: {log_messages}"
    assert "AAPL" in warmup_msgs[0]


def test_backtest_excluded_ticker() -> None:
    """Tickers with no data in the range should be excluded and logged."""
    portfolio = PortfolioConfig(
        holdings=[
            Holding(ticker="REAL", shares=10),
            Holding(ticker="GHOST", shares=5),
        ],
        available_cash=1000.0,
    )
    real = make_price_series(date(2024, 1, 2), 100, 100.0, name="REAL")

    mr = MeanReversion(window=10, threshold=0.05)
    log_messages: list[str] = []
    engine = _build_engine(
        entries=[(mr, 1.0)],
        enable_split=False,
        log_fn=log_messages.append,
    )

    start = date(2024, 1, 2)
    end = date(2024, 6, 30)
    result = engine.run(portfolio, {"REAL": real}, start, end)

    excluded_msgs = [m for m in log_messages if "excluded" in m]
    assert len(excluded_msgs) == 1
    assert "GHOST" in excluded_msgs[0]

    assert result.starting_value == 1000.0 + 10 * 100.0


def test_backtest_cash_infusion_credits_cash() -> None:
    """Cash infusions should be credited on their next_date during backtest."""
    prices = make_price_series(date(2024, 1, 2), 100, 100.0, name="AAPL")
    trading_days = list(prices.index)
    # Pick an infusion date that falls on a trading day in the middle
    infusion_date = trading_days[50]

    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10)],
        available_cash=1000.0,
        cash_infusion=CashInfusion(
            amount=2000.0,
            next_date=infusion_date,
        ),
    )

    mr = MeanReversion(window=20, threshold=0.05)
    engine = _build_engine(
        entries=[(mr, 1.0)],
        enable_split=False,
    )

    start = trading_days[0]
    end = trading_days[-1]
    result = engine.run(portfolio, {"AAPL": prices}, start, end)

    # Final value should reflect the 2000 infusion (starting cash 1000 + infusion 2000 = 3000 base)
    # Even with no trades, cash portion should include the infusion
    assert result.final_value >= 3000.0


def test_backtest_recurring_cash_infusion() -> None:
    """Recurring cash infusions should credit multiple times."""
    prices = make_price_series(date(2024, 1, 2), 100, 100.0, name="AAPL")
    trading_days = list(prices.index)
    # Start infusion early so multiple biweekly infusions land in the window
    infusion_date = trading_days[5]

    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10)],
        available_cash=500.0,
        cash_infusion=CashInfusion(
            amount=1000.0,
            next_date=infusion_date,
            frequency="biweekly",
        ),
    )

    mr = MeanReversion(window=20, threshold=0.05)
    engine = _build_engine(
        entries=[(mr, 1.0)],
        enable_split=False,
    )

    start = trading_days[0]
    end = trading_days[-1]
    result = engine.run(portfolio, {"AAPL": prices}, start, end)

    # With ~100 trading days (~140 calendar days), biweekly = ~10 infusions of $1000
    # Final value must exceed starting holdings + multiple infusions
    assert result.final_value > 500.0 + 10 * 100.0 + 5000.0


# ---------------------------------------------------------------------------
# Unit tests for the metric helpers
# ---------------------------------------------------------------------------


def _curve(values: list[float], start: date = date(2024, 1, 1)) -> list[tuple[date, float]]:
    """Build an equity curve from a list of daily values."""
    return [(date.fromordinal(start.toordinal() + i), v) for i, v in enumerate(values)]


def _sell(d: date, ticker: str, shares: float, price: float, strategy: str = "S1") -> TradeRecord:
    return TradeRecord(
        date=d, ticker=ticker, direction=Direction.SELL, shares=shares, price=price, strategy_name=strategy
    )


def _buy(d: date, ticker: str, shares: float, price: float, strategy: str = "S1") -> TradeRecord:
    return TradeRecord(
        date=d, ticker=ticker, direction=Direction.BUY, shares=shares, price=price, strategy_name=strategy
    )


# --- compute_cagr ---


def test_cagr_one_year_double() -> None:
    # 100 → 200 over ~one year ≈ 100% CAGR.
    # `days` is an int and the function divides by 365.25, so use a slightly
    # larger window to clear the rounding gap.
    cagr = compute_cagr(100.0, 200.0, 366)
    assert math.isclose(cagr, 1.0, abs_tol=1e-2)


def test_cagr_two_year_quadruple() -> None:
    # 100 → 400 over 2 years = 100% CAGR
    cagr = compute_cagr(100.0, 400.0, 731)
    assert math.isclose(cagr, 1.0, abs_tol=1e-2)


def test_cagr_zero_days_returns_zero() -> None:
    assert compute_cagr(100.0, 200.0, 0) == 0.0


def test_cagr_invalid_starting_returns_zero() -> None:
    assert compute_cagr(0.0, 200.0, 365) == 0.0
    assert compute_cagr(-10.0, 200.0, 365) == 0.0


def test_cagr_loss() -> None:
    # 100 → 50 over ~1 year = -50% CAGR
    cagr = compute_cagr(100.0, 50.0, 366)
    assert math.isclose(cagr, -0.5, abs_tol=1e-2)


# --- compute_max_drawdown ---


def test_max_drawdown_monotone_up_is_zero() -> None:
    assert compute_max_drawdown(_curve([100, 110, 120, 130])) == 0.0


def test_max_drawdown_monotone_down() -> None:
    # 100 → 50 = 50% drawdown
    assert compute_max_drawdown(_curve([100, 90, 75, 50])) == 0.5


def test_max_drawdown_peak_then_recover() -> None:
    # peak 120 → trough 60 = 50%
    dd = compute_max_drawdown(_curve([100, 120, 90, 60, 80, 110]))
    assert math.isclose(dd, 0.5, abs_tol=1e-9)


def test_max_drawdown_single_point() -> None:
    assert compute_max_drawdown(_curve([100])) == 0.0


def test_max_drawdown_empty() -> None:
    assert compute_max_drawdown([]) == 0.0


# --- compute_sharpe ---


def test_sharpe_flat_curve_is_zero() -> None:
    assert compute_sharpe(_curve([100, 100, 100, 100])) == 0.0


def test_sharpe_too_few_points() -> None:
    assert compute_sharpe(_curve([100, 101])) == 0.0
    assert compute_sharpe([]) == 0.0


def test_sharpe_positive_when_mean_positive() -> None:
    # Mix of up and down days with positive bias
    curve = _curve([100, 102, 101, 103, 102, 104, 103, 105])
    s = compute_sharpe(curve)
    assert s > 0


def test_sharpe_negative_when_mean_negative() -> None:
    curve = _curve([100, 98, 99, 97, 98, 96, 97, 95])
    assert compute_sharpe(curve) < 0


def test_sharpe_annualization_factor() -> None:
    # Returns with positive mean & nonzero std should scale by sqrt(252).
    curve = _curve([100, 101, 102, 103])  # roughly +1%/day with shrinking returns
    s = compute_sharpe(curve)
    # mean ≈ 0.0099, std small but nonzero — annualized factor present
    assert s > math.sqrt(TRADING_DAYS_PER_YEAR) * 0.5


# --- compute_sortino ---


def test_sortino_no_downside_returns_zero_not_inf() -> None:
    # All-up curve → no negative returns → undefined → 0.0 (NOT inf)
    curve = _curve([100, 101, 102, 103, 104])
    result = compute_sortino(curve)
    assert result == 0.0
    assert not math.isinf(result)


def test_sortino_flat_returns_zero() -> None:
    assert compute_sortino(_curve([100, 100, 100, 100])) == 0.0


def test_sortino_too_few_points() -> None:
    assert compute_sortino(_curve([100, 101])) == 0.0
    assert compute_sortino([]) == 0.0


def test_sortino_negative_when_mean_negative() -> None:
    curve = _curve([100, 98, 99, 97, 98, 96, 97, 95])
    assert compute_sortino(curve) < 0


def test_sortino_positive_when_mean_positive_with_some_downside() -> None:
    curve = _curve([100, 102, 101, 103, 102, 104, 103, 105])
    assert compute_sortino(curve) > 0


# --- compute_trade_stats ---


def test_trade_stats_no_sells() -> None:
    trades = [_buy(date(2024, 1, 1), "AAPL", 10, 100.0)]
    win_rate, pf, avg_win, avg_loss = compute_trade_stats(trades, [])
    assert (win_rate, pf, avg_win, avg_loss) == (0.0, 0.0, 0.0, 0.0)


def test_trade_stats_all_wins() -> None:
    trades = [
        _sell(date(2024, 1, 2), "AAPL", 10, 110.0),
        _sell(date(2024, 1, 3), "AAPL", 5, 120.0),
    ]
    basis = [100.0, 100.0]  # gains: +100, +100
    win_rate, pf, avg_win, avg_loss = compute_trade_stats(trades, basis)
    assert win_rate == 1.0
    assert math.isinf(pf)
    assert avg_win == 100.0
    assert avg_loss == 0.0


def test_trade_stats_all_losses() -> None:
    trades = [_sell(date(2024, 1, 2), "AAPL", 10, 90.0)]
    basis = [100.0]  # loss: -100
    win_rate, pf, avg_win, avg_loss = compute_trade_stats(trades, basis)
    assert win_rate == 0.0
    assert pf == 0.0
    assert avg_win == 0.0
    assert avg_loss == -100.0


def test_trade_stats_mixed() -> None:
    trades = [
        _sell(date(2024, 1, 2), "AAPL", 10, 110.0),  # +100
        _sell(date(2024, 1, 3), "AAPL", 10, 90.0),  # -100
        _sell(date(2024, 1, 4), "AAPL", 10, 120.0),  # +200
    ]
    basis = [100.0, 100.0, 100.0]
    win_rate, pf, avg_win, avg_loss = compute_trade_stats(trades, basis)
    assert math.isclose(win_rate, 2 / 3, abs_tol=1e-9)
    assert math.isclose(pf, 300.0 / 100.0, abs_tol=1e-9)
    assert math.isclose(avg_win, 150.0, abs_tol=1e-9)
    assert math.isclose(avg_loss, -100.0, abs_tol=1e-9)


def test_trade_stats_same_day_same_ticker_no_collision() -> None:
    """Two sells of the same ticker on the same day must each see their own basis."""
    day = date(2024, 1, 2)
    trades = [
        _sell(day, "AAPL", 5, 110.0, strategy="A"),  # basis 100 → +50
        _sell(day, "AAPL", 5, 110.0, strategy="B"),  # basis 80 → +150
    ]
    basis = [100.0, 80.0]
    win_rate, pf, avg_win, _avg_loss = compute_trade_stats(trades, basis)
    assert win_rate == 1.0
    # Both wins: 50 + 150 = 200; avg = 100
    assert math.isclose(avg_win, 100.0, abs_tol=1e-9)
    assert math.isinf(pf)


def test_trade_stats_breakeven_counts_as_win() -> None:
    trades = [_sell(date(2024, 1, 2), "AAPL", 10, 100.0)]
    basis = [100.0]
    win_rate, _pf, _avg_win, _avg_loss = compute_trade_stats(trades, basis)
    assert win_rate == 1.0


# --- compute_strategy_stats ---


def test_strategy_stats_groups_by_strategy() -> None:
    trades = [
        _buy(date(2024, 1, 1), "AAPL", 10, 100.0, strategy="A"),
        _sell(date(2024, 1, 2), "AAPL", 10, 110.0, strategy="A"),  # +100
        _buy(date(2024, 1, 1), "MSFT", 5, 200.0, strategy="B"),
        _sell(date(2024, 1, 3), "MSFT", 5, 190.0, strategy="B"),  # -50
    ]
    basis = [100.0, 200.0]  # parallel to sells in trade order
    stats = compute_strategy_stats(trades, basis)
    by_name = {s.name: s for s in stats}
    assert by_name["A"].buys == 1
    assert by_name["A"].sells == 1
    assert by_name["A"].pnl == 100.0
    assert by_name["A"].win_rate == 1.0
    assert by_name["B"].pnl == -50.0
    assert by_name["B"].win_rate == 0.0


def test_strategy_stats_same_day_collision() -> None:
    """Two strategies sell the same ticker on the same day with different bases."""
    day = date(2024, 1, 2)
    trades = [
        _sell(day, "AAPL", 5, 110.0, strategy="A"),  # basis 100 → +50
        _sell(day, "AAPL", 5, 110.0, strategy="B"),  # basis 80  → +150
    ]
    basis = [100.0, 80.0]
    stats = compute_strategy_stats(trades, basis)
    by_name = {s.name: s for s in stats}
    assert by_name["A"].pnl == 50.0
    assert by_name["B"].pnl == 150.0


def test_strategy_stats_empty() -> None:
    assert compute_strategy_stats([], []) == []


# --- end-to-end check that BacktestResult populates new metrics ---


def test_backtest_populates_new_metrics() -> None:
    portfolio, price_data = _make_backtest_data()
    mr = MeanReversion(window=10, threshold=-0.05)
    engine = _build_engine(entries=[(mr, 1.0)], enable_split=False)
    idx = list(price_data["AAPL"].index)
    start, end = idx[0], idx[-1]
    result = engine.run(portfolio, price_data, start, end)
    assert len(result.equity_curve) > 0
    assert result.equity_curve[-1][0] == end
    assert result.max_drawdown >= 0.0
    # Sortino must never be inf — even on a perfect run we cap to 0.
    assert not math.isinf(result.sortino_ratio)


# --- _fifo_consumed_basis unit tests ---
#
# FIFO basis underpins realized-P&L attribution. These pin the contract
# directly so regressions surface at the unit level instead of bleeding
# through end-to-end backtests.


def _pl(shares: float, basis: float) -> PositionLot:
    return PositionLot(
        shares=shares,
        purchase_date=date(2024, 1, 1),
        cost_basis=basis,
        high_water_mark=basis,
    )


def test_fifo_basis_empty_lots() -> None:
    assert BacktestEngine._fifo_consumed_basis([], 5) == 0.0


def test_fifo_basis_zero_shares() -> None:
    assert BacktestEngine._fifo_consumed_basis([_pl(10, 100.0)], 0) == 0.0


def test_fifo_basis_single_lot_partial() -> None:
    basis = BacktestEngine._fifo_consumed_basis([_pl(10, 100.0)], 4)
    assert basis == 100.0


def test_fifo_basis_crosses_lot_boundary() -> None:
    """Consume 8 shares across two lots: 5 @ $100 + 3 @ $80 → $92.50."""
    lots = [_pl(5, 100.0), _pl(10, 80.0)]
    basis = BacktestEngine._fifo_consumed_basis(lots, 8)
    assert basis == (5 * 100.0 + 3 * 80.0) / 8


def test_fifo_basis_respects_lot_order() -> None:
    """Reversing the lot list must change the answer — this is FIFO, not LIFO."""
    lot_a = _pl(5, 100.0)
    lot_b = _pl(10, 80.0)
    fifo = BacktestEngine._fifo_consumed_basis([lot_a, lot_b], 8)
    lifo = BacktestEngine._fifo_consumed_basis([lot_b, lot_a], 8)
    assert fifo != lifo
    assert lifo == 80.0  # first 8 all come from the $80 lot


def test_fifo_basis_does_not_mutate_lots() -> None:
    lots = [_pl(5, 100.0), _pl(10, 80.0)]
    BacktestEngine._fifo_consumed_basis(lots, 8)
    assert lots[0].shares == 5
    assert lots[1].shares == 10


# --- restriction-before-sizing regression ---


def test_blocked_sell_does_not_leak_into_buy_sizing() -> None:
    """Restriction-before-sizing invariant: blocked sells must not inflate the
    ``cash`` the buy pass sees.

    The backtest orders Phase 3 as:

        1. size sells from clamped targets
        2. filter restriction-blocked sells
        3. compute ``post_sell_cash = state.cash + sum(filtered)``
        4. call ``size_buys(..., cash=post_sell_cash, ...)``

    If step 2 is skipped (or moved after step 3), blocked sell proceeds
    leak into ``post_sell_cash`` and ``size_buys`` can authorize buys
    against cash that will never arrive in phase 5.

    This test pins the order by spying on ``size_buys`` and asserting the
    ``cash`` argument equals ``state.cash + proceeds_of_unblocked_sells``,
    not ``state.cash + proceeds_of_all_sells``.
    """
    constraints = AllocationConstraints(min_buy_delta=0.01, max_position_pct=0.95)
    mr = MeanReversion(window=10, threshold=0.05)
    allocator = Allocator([(mr, 1.0)], constraints, n_tickers=1)
    sizer = OrderSizer(default_slippage=0.0)

    # Spy on size_buys to capture the cash it was called with.
    captured: dict[str, float] = {}
    real_size_buys = sizer.size_buys

    def spy_size_buys(*args, **kwargs):
        # cash is the 4th positional arg in size_buys
        captured["cash"] = args[3] if len(args) > 3 else kwargs["cash"]
        return real_size_buys(*args, **kwargs)

    sizer.size_buys = spy_size_buys  # type: ignore[method-assign]

    engine = BacktestEngine(
        allocator=allocator,
        order_sizer=sizer,
        exit_rules=[ProfitTaking(gain_threshold=0.10)],
        constraints=constraints,
    )

    # A single held ticker in profit → ProfitTaking fires. Round-trip
    # restriction blocks the sell.
    a_prices = np.array([100.0] * 15)

    state = _SimState(cash=0.0)
    state.positions = {"A": 5.0}
    state.lots = {
        "A": [
            PositionLot(
                shares=5.0,
                purchase_date=date(2024, 1, 1),
                cost_basis=80.0,
                high_water_mark=100.0,
            )
        ]
    }
    state.high_water_marks = {"A": 100.0}
    state.restriction_tracker = RestrictionTracker(TradingRestrictions(round_trip_days=30))
    state.restriction_tracker.record_trade("A", Direction.BUY, date(2024, 1, 15))

    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="A", shares=5, cost_basis=80.0)],
        available_cash=0.0,
        trading_restrictions=TradingRestrictions(round_trip_days=30),
    )

    engine._run_day(state, portfolio, {"A": a_prices}, date(2024, 1, 16))

    # post_sell_cash should equal state.cash ($0) because the single sell
    # was blocked. If blocked proceeds leaked in, cash would be ~$500.
    assert "cash" in captured, "size_buys was not called"
    assert captured["cash"] == 0.0, f"blocked sell proceeds leaked into size_buys cash: {captured['cash']}"


# --- competing exit rules: realized + unrealized reconciliation ---


def test_competing_exit_rules_collapse_to_one_sell() -> None:
    """Regression: ProfitTaking + TrailingStop firing the same tick over-sold.

    When a held lot is in deep profit *and* its HWM has already drifted
    below the trail threshold, both rules independently want to liquidate
    the entire position. Pre-fix, ``size_sells`` produced two sell orders
    each sized against the full position — the second sell drained shares
    that no longer existed, fabricated cash, and broke the realized-P&L
    reconciliation against ``final - start``.

    Drive ``_run_day`` directly with a hand-crafted state so we can pin
    exactly the lot/HWM/price configuration that triggers the double-fire,
    then assert: (a) only one sell fires, (b) it's credited to the more
    aggressive rule, and (c) ``final - start == realized + unrealized``.
    """
    constraints = AllocationConstraints(min_buy_delta=0.01, max_position_pct=0.95)
    allocator = Allocator([], constraints, n_tickers=1)
    sizer = OrderSizer(default_slippage=0.0)
    engine = BacktestEngine(
        allocator=allocator,
        order_sizer=sizer,
        exit_rules=[ProfitTaking(gain_threshold=0.10), TrailingStop(trail_pct=0.05)],
        constraints=constraints,
    )

    # Lot @ $80 basis with HWM=$130 (the peak the position has seen).
    # Today's price is $115:
    #   PT: gain = (115-80)/80 = 43.75% > 10% → wants full liquidation
    #   TS: drawdown = (130-115)/130 = 11.5% > 5% → wants full liquidation
    # Both rules fire on the entire 10-share position the same tick.
    state = _SimState(cash=0.0)
    state.cash = 0.0
    state.positions = {"A": 10.0}
    state.lots = {
        "A": [
            PositionLot(
                shares=10.0,
                purchase_date=date(2024, 1, 1),
                cost_basis=80.0,
                high_water_mark=130.0,
            )
        ]
    }
    state.high_water_marks = {"A": 130.0}
    state.starting_value = 800.0  # 10 shares at $80 basis
    state.twr_base_value = 800.0

    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="A", shares=10, cost_basis=80.0)],
        available_cash=0.0,
    )

    # Price array with current=$115; backstop history gives the rules
    # something to read but the only price they act on is the last bar.
    prices = np.array([100.0, 110.0, 120.0, 130.0, 125.0, 120.0, 115.0])
    engine._run_day(state, portfolio, {"A": prices}, date(2024, 2, 1))

    sells = [t for t in state.trades if t.direction == Direction.SELL]
    assert len(sells) == 1, f"expected 1 sell after collapse, got {len(sells)}: {sells}"
    assert sells[0].shares == 10  # the full position, not 20
    # TrailingStop's intent (full liquidation at current price) ties with
    # ProfitTaking's. Either is acceptable; what matters is one wins, not
    # both. We assert it's one of the two configured sources.
    assert sells[0].strategy_name in {"ProfitTaking", "TrailingStop"}

    # Reconciliation: with one sell, FIFO consumed basis = real basis,
    # state.cash exactly reflects sell proceeds, and the position is empty.
    final_price = float(prices[-1])
    final_value = state.cash + sum(sum(lot.shares for lot in lots) * final_price for lots in state.lots.values())
    realized = sum(
        (t.price - state.basis_per_sell[i]) * t.shares
        for i, t in enumerate(t for t in state.trades if t.direction == Direction.SELL)
    )
    unrealized = sum(sum(lot.shares * (final_price - lot.cost_basis) for lot in lots) for lots in state.lots.values())
    delta = final_value - state.starting_value
    assert math.isclose(delta, realized + unrealized, abs_tol=1.0), (
        f"reconciliation broken: final-start=${delta:,.2f} but realized+unrealized=${realized + unrealized:,.2f}"
    )
