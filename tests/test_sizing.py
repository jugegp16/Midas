"""Tests for the order sizing engine."""

from datetime import UTC, date, datetime

from midas.models import CashInfusion, Direction, Signal
from midas.sizing import SizingConfig, SizingEngine


def _make_signal(
    ticker: str = "AAPL",
    direction: Direction = Direction.BUY,
    price: float = 100.0,
) -> Signal:
    return Signal(
        ticker=ticker,
        direction=direction,
        strength=0.5,
        reasoning="test",
        timestamp=datetime.now(tz=UTC),
        price=price,
        strategy_name="Test",
    )


class TestBuySizing:
    def test_whole_shares_only(self) -> None:
        config = SizingConfig(default_slippage=0.0, circuit_breaker_pct=1.0)
        engine = SizingEngine(config)
        signal = _make_signal(price=150.0)
        order = engine.size_order(signal, available_cash=400.0, position_shares=0)
        assert order is not None
        assert order.shares == 2  # 400 / 150 = 2.66 -> 2
        assert order.direction == Direction.BUY

    def test_slippage_reduces_shares(self) -> None:
        engine = SizingEngine(SizingConfig(default_slippage=0.10))  # 10% slippage
        signal = _make_signal(price=100.0)
        # Effective price = 110, cash = 500, circuit breaker = 25% of 500 = 125
        # shares = floor(125 / 110) = 1
        order = engine.size_order(signal, available_cash=500.0, position_shares=0)
        assert order is not None
        assert order.shares == 1

    def test_circuit_breaker(self) -> None:
        config = SizingConfig(circuit_breaker_pct=0.25, default_slippage=0.0)
        engine = SizingEngine(config)
        signal = _make_signal(price=100.0)
        # Cash = 1000, max today = 250, already deployed 250 -> 0 remaining
        order = engine.size_order(
            signal, available_cash=1000.0, position_shares=0, daily_deployed=250.0
        )
        assert order is not None
        assert order.shares == 0

    def test_pending_cash_infusion(self) -> None:
        config = SizingConfig(
            default_slippage=0.0,
            circuit_breaker_pct=1.0,  # disable breaker
            infusion_lookahead_days=3,
        )
        engine = SizingEngine(config)
        signal = _make_signal(price=100.0)
        infusion = CashInfusion(amount=500.0, next_date=date(2025, 1, 3))
        order = engine.size_order(
            signal,
            available_cash=200.0,
            position_shares=0,
            cash_infusion=infusion,
            today=date(2025, 1, 1),
        )
        assert order is not None
        assert order.shares == 7  # (200 + 500) / 100 = 7
        assert order.relies_on_pending_cash is True

    def test_no_pending_cash_when_too_far(self) -> None:
        config = SizingConfig(
            default_slippage=0.0,
            circuit_breaker_pct=1.0,
            infusion_lookahead_days=3,
        )
        engine = SizingEngine(config)
        signal = _make_signal(price=100.0)
        infusion = CashInfusion(amount=500.0, next_date=date(2025, 1, 15))
        order = engine.size_order(
            signal,
            available_cash=200.0,
            position_shares=0,
            cash_infusion=infusion,
            today=date(2025, 1, 1),
        )
        assert order is not None
        assert order.shares == 2  # 200 / 100 = 2
        assert order.relies_on_pending_cash is False


class TestSellSizing:
    def test_sells_full_position(self) -> None:
        engine = SizingEngine(SizingConfig(default_slippage=0.0))
        signal = _make_signal(direction=Direction.SELL, price=150.0)
        order = engine.size_order(signal, available_cash=0, position_shares=10)
        assert order is not None
        assert order.shares == 10
        assert order.direction == Direction.SELL

    def test_zero_position_filtered(self) -> None:
        engine = SizingEngine()
        signal = _make_signal(direction=Direction.SELL, price=150.0)
        order = engine.size_order(signal, available_cash=0, position_shares=0)
        assert order is None
