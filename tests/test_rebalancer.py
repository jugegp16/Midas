"""Tests for the Rebalancer."""

from __future__ import annotations

from midas.allocator import AllocationResult
from midas.models import (
    AllocationConstraints,
    Direction,
    MechanicalIntent,
)
from midas.rebalancer import Rebalancer, RebalancerConfig


def _alloc_result(
    targets: dict[str, float],
    blended: dict[str, float] | None = None,
    contribs: dict[str, dict[str, float]] | None = None,
    trim_reasons: dict[str, str] | None = None,
) -> AllocationResult:
    blended = blended or {t: 0.0 for t in targets}
    contribs = contribs or {t: {} for t in targets}
    trim_reasons = trim_reasons or {}
    return AllocationResult(targets, contribs, blended, trim_reasons)


class TestRebalancer:
    def test_no_rebalance_within_threshold(self):
        """Positions within rebalance_threshold generate no orders."""
        r = Rebalancer()
        allocation = _alloc_result({"A": 0.50})
        # Current position is 50% of portfolio -> no delta
        positions = {"A": 50.0}
        prices = {"A": 100.0}
        cash = 5000.0  # total = 5000 + 5000 = 10000, A = 50%
        constraints = AllocationConstraints(rebalance_threshold=0.02)
        orders = r.generate_orders(allocation, positions, prices, cash, constraints)
        assert orders == []

    def test_sell_generated_when_overweight(self):
        """Should generate SELL when current > target by more than threshold."""
        r = Rebalancer(RebalancerConfig(default_slippage=0.0))
        allocation = _alloc_result({"A": 0.30}, contribs={"A": {"ProfitTaking": -0.5}})
        # A is 50% of portfolio, target is 30% -> need to sell
        positions = {"A": 50.0}
        prices = {"A": 100.0}
        cash = 5000.0  # total = 10000, A weight = 0.50
        constraints = AllocationConstraints(rebalance_threshold=0.02)
        orders = r.generate_orders(allocation, positions, prices, cash, constraints)
        assert len(orders) == 1
        assert orders[0].direction == Direction.SELL
        assert orders[0].ticker == "A"
        # Should sell ~20% of portfolio = $2000 = 20 shares
        assert orders[0].shares == 20

    def test_buy_generated_when_underweight(self):
        """Should generate BUY when current < target by more than threshold."""
        r = Rebalancer(RebalancerConfig(default_slippage=0.0))
        allocation = _alloc_result({"A": 0.50}, contribs={"A": {"Momentum": 0.5}})
        # A is 30% of portfolio, target is 50% -> need to buy
        positions = {"A": 30.0}
        prices = {"A": 100.0}
        cash = 7000.0  # total = 10000, A weight = 0.30
        constraints = AllocationConstraints(rebalance_threshold=0.02)
        orders = r.generate_orders(allocation, positions, prices, cash, constraints)
        assert len(orders) == 1
        assert orders[0].direction == Direction.BUY
        assert orders[0].ticker == "A"
        # Should buy ~20% of portfolio = $2000 = 20 shares
        assert orders[0].shares == 20

    def test_sells_before_buys(self):
        """Sells should appear before buys in the order list."""
        r = Rebalancer(RebalancerConfig(default_slippage=0.0))
        allocation = _alloc_result(
            {"A": 0.20, "B": 0.60},
            contribs={"A": {"ProfitTaking": -0.5}, "B": {"Momentum": 0.5}},
        )
        # A overweight (50%), B underweight (20%)
        positions = {"A": 50.0, "B": 20.0}
        prices = {"A": 100.0, "B": 100.0}
        cash = 3000.0  # total = 10000
        constraints = AllocationConstraints(rebalance_threshold=0.02)
        orders = r.generate_orders(allocation, positions, prices, cash, constraints)
        sell_idx = [i for i, o in enumerate(orders) if o.direction == Direction.SELL]
        buy_idx = [i for i, o in enumerate(orders) if o.direction == Direction.BUY]
        if sell_idx and buy_idx:
            assert max(sell_idx) < min(buy_idx)

    def test_cash_constraint_limits_buys(self):
        """Buy orders are reduced when cash is insufficient."""
        r = Rebalancer(RebalancerConfig(default_slippage=0.0))
        allocation = _alloc_result({"A": 0.90}, contribs={"A": {"Momentum": 0.9}})
        positions = {"A": 0.0}
        prices = {"A": 100.0}
        cash = 500.0  # total = 500, want to buy 90% = $450 = 4 shares
        constraints = AllocationConstraints(rebalance_threshold=0.02)
        orders = r.generate_orders(allocation, positions, prices, cash, constraints)
        assert len(orders) == 1
        assert orders[0].shares <= 5  # limited by cash

    def test_circuit_breaker_limits_deployment(self):
        """Circuit breaker caps total buy deployment per day."""
        r = Rebalancer(RebalancerConfig(default_slippage=0.0, circuit_breaker_pct=0.10))
        allocation = _alloc_result({"A": 0.90}, contribs={"A": {"Momentum": 0.9}})
        positions = {"A": 0.0}
        prices = {"A": 10.0}
        cash = 10000.0  # total = 10000
        constraints = AllocationConstraints(rebalance_threshold=0.02)
        orders = r.generate_orders(allocation, positions, prices, cash, constraints)
        total_deployed = sum(o.estimated_value for o in orders if o.direction == Direction.BUY)
        # Circuit breaker = 10% of 10000 = 1000
        assert total_deployed <= 1000.0 + 1.0  # small tolerance for rounding

    def test_slippage_applied(self):
        """Slippage adjusts execution prices."""
        r = Rebalancer(RebalancerConfig(default_slippage=0.01))
        allocation = _alloc_result({"A": 0.80}, contribs={"A": {"Momentum": 0.8}})
        positions = {"A": 0.0}
        prices = {"A": 100.0}
        cash = 10000.0
        constraints = AllocationConstraints(rebalance_threshold=0.02)
        orders = r.generate_orders(allocation, positions, prices, cash, constraints)
        assert len(orders) >= 1
        buy = orders[0]
        assert buy.price > 100.0  # slippage raises buy price

    def test_fallback_source_tagged_with_cap(self):
        """When no strategy drove the trade and trim_reason=cap, source = Rebalancer (cap)."""
        r = Rebalancer(RebalancerConfig(default_slippage=0.0))
        # No contribs → fallback path. Cap trim recorded.
        allocation = _alloc_result(
            {"A": 0.10},
            contribs={"A": {}},
            trim_reasons={"A": "cap"},
        )
        positions = {"A": 50.0}  # currently 50%
        prices = {"A": 100.0}
        cash = 5000.0  # total = 10000
        constraints = AllocationConstraints(rebalance_threshold=0.02)
        orders = r.generate_orders(allocation, positions, prices, cash, constraints)
        assert len(orders) == 1
        assert orders[0].context.source == "Rebalancer (cap)"

    def test_unjustified_trade_skipped(self):
        """No aligned contrib AND no trim reason → order is suppressed entirely.

        Unjustified drift-correction trades serve no purpose and should not
        appear in the order book. Softmax + Option A eliminate most of these
        at the source; this check is a belt-and-braces guard.
        """
        r = Rebalancer(RebalancerConfig(default_slippage=0.0))
        allocation = _alloc_result({"A": 0.30}, contribs={"A": {}})
        positions = {"A": 50.0}
        prices = {"A": 100.0}
        cash = 5000.0
        constraints = AllocationConstraints(rebalance_threshold=0.02)
        orders = r.generate_orders(allocation, positions, prices, cash, constraints)
        assert orders == []

    def test_order_context_populated(self):
        """Orders carry proper OrderContext."""
        r = Rebalancer(RebalancerConfig(default_slippage=0.0))
        allocation = _alloc_result(
            {"A": 0.60},
            blended={"A": 0.5},
            contribs={"A": {"MeanReversion": 0.5}},
        )
        positions = {"A": 10.0}
        prices = {"A": 100.0}
        cash = 5000.0
        constraints = AllocationConstraints(rebalance_threshold=0.02)
        orders = r.generate_orders(allocation, positions, prices, cash, constraints)
        assert len(orders) >= 1
        ctx = orders[0].context
        assert ctx.blended_score == 0.5
        assert ctx.target_weight == 0.60
        assert "MeanReversion" in ctx.contributions


class TestSizeMechanical:
    def test_dca_buy_intent(self):
        """Mechanical buy intent is sized into an order."""
        r = Rebalancer(RebalancerConfig(default_slippage=0.0))
        intent = MechanicalIntent(
            ticker="VOO",
            direction=Direction.BUY,
            target_value=500.0,
            reason="DCA trigger",
            source="DollarCostAveraging",
        )
        orders = r.size_mechanical([intent], cash=1000.0, prices={"VOO": 100.0})
        assert len(orders) == 1
        assert orders[0].ticker == "VOO"
        assert orders[0].direction == Direction.BUY
        assert orders[0].shares == 5

    def test_insufficient_cash_limits_mechanical(self):
        """Mechanical buy limited by available cash."""
        r = Rebalancer(RebalancerConfig(default_slippage=0.0))
        intent = MechanicalIntent(
            ticker="VOO",
            direction=Direction.BUY,
            target_value=5000.0,
            reason="DCA trigger",
            source="DollarCostAveraging",
        )
        orders = r.size_mechanical([intent], cash=200.0, prices={"VOO": 100.0})
        assert len(orders) == 1
        assert orders[0].shares == 2
