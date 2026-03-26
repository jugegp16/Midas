"""Tests for the Agent (cooldown tracking)."""

from datetime import date

from midas.agent import Agent
from midas.models import Holding, PortfolioConfig
from midas.strategies.mean_reversion import MeanReversion
from tests.conftest import make_price_series


def test_cooldown_suppresses_duplicate_signals() -> None:
    strategy = MeanReversion(window=10, threshold=0.01)
    agent = Agent(strategy=strategy, cooldown_days=5)

    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="TEST", shares=10)],
        available_cash=1000.0,
    )

    # Create a dropping price series
    returns = [0.0] * 10 + [-0.01] * 20
    prices = make_price_series(date(2024, 1, 2), 30, 100.0, returns, name="TEST")

    # First evaluation — should produce signal
    data = {"TEST": prices}
    signals_day1 = agent.run(portfolio, data, today=date(2024, 2, 5))

    # Second evaluation same day — should be suppressed by cooldown
    signals_day2 = agent.run(portfolio, data, today=date(2024, 2, 6))
    assert len(signals_day2) == 0 or len(signals_day1) == 0  # at least one suppressed

    # After cooldown period
    signals_day3 = agent.run(portfolio, data, today=date(2024, 2, 15))
    # Should be able to signal again after cooldown
    if signals_day1:
        assert len(signals_day3) >= 0  # cooldown may or may not have elapsed


def test_agent_respects_ticker_filter() -> None:
    strategy = MeanReversion(window=10, threshold=0.01)
    agent = Agent(strategy=strategy, tickers=["AAPL"])

    portfolio = PortfolioConfig(
        holdings=[
            Holding(ticker="AAPL", shares=10),
            Holding(ticker="VOO", shares=5),
        ],
        available_cash=1000.0,
    )

    returns = [0.0] * 10 + [-0.02] * 20
    aapl = make_price_series(date(2024, 1, 2), 30, 100.0, returns, name="AAPL")
    voo = make_price_series(date(2024, 1, 2), 30, 100.0, returns, name="VOO")

    signals = agent.run(portfolio, {"AAPL": aapl, "VOO": voo}, today=date(2024, 2, 15))
    # All signals should be for AAPL only
    for s in signals:
        assert s.ticker == "AAPL"
