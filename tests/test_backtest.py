"""Tests for the backtest engine."""

import math
from datetime import date
from pathlib import Path

import pandas as pd
from conftest import make_price_series

from midas.allocator import Allocator
from midas.backtest import (
    TRADING_DAYS_PER_YEAR,
    BacktestEngine,
    compute_cagr,
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
    compute_strategy_stats,
    compute_trade_stats,
    write_backtest_csv,
)
from midas.models import AllocationConstraints, CashInfusion, Direction, Holding, PortfolioConfig, TradeRecord
from midas.order_sizer import OrderSizer
from midas.strategies.mean_reversion import MeanReversion
from midas.strategies.profit_taking import ProfitTaking


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
