"""Tests for the backtest engine."""

from datetime import date
from pathlib import Path

import pandas as pd
from conftest import make_price_series

from midas.allocator import Allocator
from midas.backtest import BacktestEngine, write_backtest_csv
from midas.models import AllocationConstraints, CashInfusion, Direction, Holding, PortfolioConfig
from midas.rebalancer import Rebalancer
from midas.strategies.mean_reversion import MeanReversion
from midas.strategies.profit_taking import ProfitTaking


def _build_engine(
    conviction_strategies=None,
    protective_strategies=None,
    mechanical_strategies=None,
    constraints=None,
    n_tickers=1,
    **kwargs,
):
    """Helper to build a BacktestEngine with the new allocator system."""
    conviction = conviction_strategies or []
    protective = protective_strategies or []
    constraints = constraints or AllocationConstraints(
        rebalance_threshold=0.01,
        max_position_pct=0.95,
    )
    allocator = Allocator(conviction, protective, constraints, n_tickers)
    rebalancer = Rebalancer()
    return BacktestEngine(
        allocator=allocator,
        rebalancer=rebalancer,
        mechanical_strategies=mechanical_strategies,
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
        conviction_strategies=[(mr, 1.0)],
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
        conviction_strategies=[(mr, 1.0)],
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
        conviction_strategies=[(mr, 1.0)],
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
        conviction_strategies=[(pt, 1.0)],
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
        conviction_strategies=[(mr, 1.0)],
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
        conviction_strategies=[(mr, 1.0)],
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
        conviction_strategies=[(mr, 1.0)],
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
        conviction_strategies=[(mr, 1.0)],
        enable_split=False,
    )

    start = trading_days[0]
    end = trading_days[-1]
    result = engine.run(portfolio, {"AAPL": prices}, start, end)

    # With ~100 trading days (~140 calendar days), biweekly = ~10 infusions of $1000
    # Final value must exceed starting holdings + multiple infusions
    assert result.final_value > 500.0 + 10 * 100.0 + 5000.0
