"""Tests for core data models."""

from datetime import UTC, date, datetime

from midas.models import (
    Direction,
    Holding,
    HoldingPeriod,
    Order,
    PortfolioConfig,
    Signal,
    TradeRecord,
)


def test_signal_creation() -> None:
    sig = Signal(
        ticker="AAPL",
        direction=Direction.BUY,
        strength=0.5,
        reasoning="test",
        timestamp=datetime.now(tz=UTC),
        price=150.0,
        strategy_name="TestStrategy",
    )
    assert sig.ticker == "AAPL"
    assert sig.direction == Direction.BUY
    assert 0.0 <= sig.strength <= 1.0


def test_portfolio_get_holding() -> None:
    port = PortfolioConfig(
        holdings=[
            Holding(ticker="AAPL", shares=10, cost_basis=150.0),
            Holding(ticker="VOO", shares=5),
        ],
        available_cash=1000.0,
    )
    assert port.get_holding("AAPL") is not None
    assert port.get_holding("AAPL").shares == 10  # type: ignore[union-attr]
    assert port.get_holding("VOO").cost_basis is None  # type: ignore[union-attr]
    assert port.get_holding("MSFT") is None


def test_holding_period_values() -> None:
    assert HoldingPeriod.SHORT_TERM.value == "short-term"
    assert HoldingPeriod.LONG_TERM.value == "long-term"


def test_order_frozen() -> None:
    sig = Signal(
        ticker="VOO",
        direction=Direction.BUY,
        strength=0.3,
        reasoning="test",
        timestamp=datetime.now(tz=UTC),
        price=500.0,
        strategy_name="Test",
    )
    order = Order(
        ticker="VOO",
        direction=Direction.BUY,
        shares=2,
        estimated_value=1000.0,
        signal=sig,
    )
    assert order.relies_on_pending_cash is False


def test_trade_record() -> None:
    tr = TradeRecord(
        date=date(2024, 6, 1),
        ticker="AAPL",
        direction=Direction.SELL,
        shares=5,
        price=180.0,
        strategy_name="ProfitTaking",
        holding_period=HoldingPeriod.LONG_TERM,
    )
    assert tr.holding_period == HoldingPeriod.LONG_TERM
