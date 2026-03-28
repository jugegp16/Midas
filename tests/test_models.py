"""Tests for core data models."""

from datetime import date

from midas.models import (
    AllocationConstraints,
    Direction,
    Holding,
    HoldingPeriod,
    MechanicalIntent,
    Order,
    OrderContext,
    PortfolioConfig,
    StrategyConfig,
    StrategyTier,
    TradeRecord,
)


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
    ctx = OrderContext(
        contributions={"TestStrategy": 0.5},
        blended_score=0.5,
        target_weight=0.10,
        current_weight=0.05,
        reason="test buy",
    )
    order = Order(
        ticker="VOO",
        direction=Direction.BUY,
        shares=2,
        price=500.0,
        estimated_value=1000.0,
        context=ctx,
    )
    assert order.context.blended_score == 0.5
    assert order.price == 500.0


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


def test_strategy_tier_values() -> None:
    assert StrategyTier.CONVICTION.value == "conviction"
    assert StrategyTier.PROTECTIVE.value == "protective"
    assert StrategyTier.MECHANICAL.value == "mechanical"


def test_allocation_constraints_defaults() -> None:
    c = AllocationConstraints()
    assert c.max_position_pct is None
    assert c.min_cash_pct == 0.05
    assert c.rebalance_threshold == 0.02
    assert c.sigmoid_steepness == 2.0


def test_mechanical_intent() -> None:
    intent = MechanicalIntent(
        ticker="VOO",
        direction=Direction.BUY,
        target_value=500.0,
        reason="DCA",
    )
    assert intent.target_value == 500.0


def test_strategy_config_defaults() -> None:
    cfg = StrategyConfig(name="TestStrategy")
    assert cfg.weight == 1.0
    assert cfg.tier == StrategyTier.CONVICTION
    assert cfg.veto_threshold == -0.5
