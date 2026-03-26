"""Shared test fixtures — synthetic price data and configs."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from midas.models import CashInfusion, Holding, PortfolioConfig


@pytest.fixture
def sample_portfolio() -> PortfolioConfig:
    return PortfolioConfig(
        holdings=[
            Holding(ticker="AAPL", shares=10, cost_basis=150.0),
            Holding(ticker="VOO", shares=5, cost_basis=400.0),
        ],
        available_cash=5000.0,
        cash_infusion=CashInfusion(
            amount=1500.0,
            next_date=date(2025, 1, 10),
        ),
    )


def make_price_series(
    start: date,
    days: int,
    base_price: float,
    daily_returns: list[float] | None = None,
    name: str = "TEST",
) -> pd.Series:
    """Generate a synthetic price series.

    If daily_returns is None, generates a flat price series.
    """
    dates = []
    prices = []
    price = base_price
    current = start
    for i in range(days):
        # Skip weekends
        while current.weekday() >= 5:
            current += timedelta(days=1)
        dates.append(current)
        if daily_returns and i < len(daily_returns):
            price *= 1 + daily_returns[i]
        prices.append(round(price, 2))
        current += timedelta(days=1)
    series = pd.Series(prices, index=dates, name=name)
    return series


@pytest.fixture
def flat_prices() -> pd.Series:
    """100 days of flat $100 price."""
    return make_price_series(date(2024, 1, 2), 100, 100.0, name="FLAT")


@pytest.fixture
def dropping_prices() -> pd.Series:
    """Price is stable then drops sharply at the end — triggers mean reversion.

    The last few days are a sharp drop while the 30-day MA still includes
    the stable period, creating a gap between current price and MA.
    """
    returns = [0.0] * 90 + [-0.02] * 10
    return make_price_series(date(2024, 1, 2), 100, 100.0, returns, name="DROP")


@pytest.fixture
def rising_prices() -> pd.Series:
    """Price rises steadily — triggers profit taking."""
    returns = [0.003] * 100
    return make_price_series(date(2024, 1, 2), 100, 100.0, returns, name="RISE")


@pytest.fixture
def crossover_prices() -> pd.Series:
    """Price dips below MA then crosses back above — triggers momentum."""
    returns = [0.0] * 20 + [-0.008] * 15 + [0.015] * 10 + [0.0] * 55
    return make_price_series(date(2024, 1, 2), 100, 100.0, returns, name="CROSS")


@pytest.fixture
def volatile_dropping_prices() -> pd.Series:
    """Price with sustained losses at the end — triggers RSI oversold."""
    returns = [0.001] * 80 + [-0.02] * 20
    return make_price_series(date(2024, 1, 2), 100, 100.0, returns, name="VDROP")


@pytest.fixture
def volatile_rising_prices() -> pd.Series:
    """Strong sustained gains at the end — triggers RSI overbought."""
    returns = [0.001] * 80 + [0.02] * 20
    return make_price_series(date(2024, 1, 2), 100, 100.0, returns, name="VRISE")


@pytest.fixture
def gap_down_recovery_prices() -> pd.Series:
    """Price stable then sharp drop then recovery — triggers gap down recovery."""
    returns = [0.0] * 95 + [-0.05, -0.04, 0.06] + [0.0] * 2
    return make_price_series(date(2024, 1, 2), 100, 100.0, returns, name="GAP")


@pytest.fixture
def peak_then_drop_prices() -> pd.Series:
    """Price rises then falls — triggers trailing stop."""
    returns = [0.01] * 30 + [-0.005] * 40 + [0.0] * 30
    return make_price_series(date(2024, 1, 2), 100, 100.0, returns, name="PEAK")


@pytest.fixture
def ma_crossover_prices() -> pd.Series:
    """Long decline followed by recovery — triggers golden cross."""
    returns = [-0.002] * 60 + [0.008] * 30 + [0.0] * 10
    return make_price_series(date(2024, 1, 2), 100, 100.0, returns, name="MACROSS")
