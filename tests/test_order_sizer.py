"""Tests for the OrderSizer."""

from __future__ import annotations

from midas.allocator import AllocationResult
from midas.models import (
    AllocationConstraints,
    Direction,
    ExitIntent,
)
from midas.order_sizer import OrderSizer


def _alloc_result(
    targets: dict[str, float],
    blended: dict[str, float] | None = None,
    contribs: dict[str, dict[str, float]] | None = None,
) -> AllocationResult:
    blended = blended or {t: 0.0 for t in targets}
    contribs = contribs or {t: {} for t in targets}
    return AllocationResult(targets, contribs, blended)


class TestSizeBuys:
    def test_no_buys_when_within_threshold(self):
        """Buy delta below ``min_buy_delta`` is suppressed."""
        sizer = OrderSizer()
        allocation = _alloc_result({"A": 0.50})
        positions = {"A": 50.0}
        prices = {"A": 100.0}
        cash = 5000.0  # total = 10000, A weight = 0.50, delta = 0
        constraints = AllocationConstraints(min_buy_delta=0.02)
        orders = sizer.size_buys(allocation, positions, prices, cash, constraints, total_value=10000.0)
        assert orders == []

    def test_no_sells_on_negative_delta(self):
        """OrderSizer is buy-only — drift above target never produces a sell."""
        sizer = OrderSizer(default_slippage=0.0)
        allocation = _alloc_result({"A": 0.30}, contribs={"A": {"Momentum": 0.5}})
        positions = {"A": 50.0}  # 50% currently, target 30% — would be a sell
        prices = {"A": 100.0}
        cash = 5000.0
        constraints = AllocationConstraints(min_buy_delta=0.02)
        orders = sizer.size_buys(allocation, positions, prices, cash, constraints, total_value=10000.0)
        assert orders == []

    def test_buy_generated_when_underweight(self):
        """Should generate BUY when current < target by more than threshold."""
        sizer = OrderSizer(default_slippage=0.0)
        allocation = _alloc_result({"A": 0.50}, contribs={"A": {"Momentum": 0.5}})
        positions = {"A": 30.0}
        prices = {"A": 100.0}
        cash = 7000.0  # total = 10000, A weight = 0.30
        constraints = AllocationConstraints(min_buy_delta=0.02)
        orders = sizer.size_buys(allocation, positions, prices, cash, constraints, total_value=10000.0)
        assert len(orders) == 1
        assert orders[0].direction == Direction.BUY
        assert orders[0].ticker == "A"
        # ~20% of $10k = $2000 = 20 shares
        assert orders[0].shares == 20

    def test_cash_constraint_limits_buys(self):
        """Buy orders are reduced when cash is insufficient."""
        sizer = OrderSizer(default_slippage=0.0)
        allocation = _alloc_result({"A": 0.90}, contribs={"A": {"Momentum": 0.9}})
        positions = {"A": 0.0}
        prices = {"A": 100.0}
        cash = 500.0
        constraints = AllocationConstraints(min_buy_delta=0.02)
        orders = sizer.size_buys(allocation, positions, prices, cash, constraints, total_value=500.0)
        assert len(orders) == 1
        assert orders[0].shares <= 5

    def test_circuit_breaker_limits_deployment(self):
        """Circuit breaker caps total buy deployment per day."""
        sizer = OrderSizer(default_slippage=0.0, circuit_breaker_pct=0.10)
        allocation = _alloc_result({"A": 0.90}, contribs={"A": {"Momentum": 0.9}})
        positions = {"A": 0.0}
        prices = {"A": 10.0}
        cash = 10000.0
        constraints = AllocationConstraints(min_buy_delta=0.02)
        orders = sizer.size_buys(allocation, positions, prices, cash, constraints, total_value=10000.0)
        total_deployed = sum(o.estimated_value for o in orders if o.direction == Direction.BUY)
        assert total_deployed <= 1000.0 + 1.0

    def test_slippage_applied(self):
        """Slippage adjusts execution prices."""
        sizer = OrderSizer(default_slippage=0.01)
        allocation = _alloc_result({"A": 0.80}, contribs={"A": {"Momentum": 0.8}})
        positions = {"A": 0.0}
        prices = {"A": 100.0}
        cash = 10000.0
        constraints = AllocationConstraints(min_buy_delta=0.02)
        orders = sizer.size_buys(allocation, positions, prices, cash, constraints, total_value=10000.0)
        assert len(orders) >= 1
        assert orders[0].price > 100.0

    def test_held_ticker_no_phantom_buy_after_sell_in_same_tick(self):
        """Regression: held tickers must not fire phantom buys when another
        ticker sold in the same tick freed cash.

        Setup mirrors what the backtest engine does:
        - Pre-sell snapshot: A=$5k held, B=$5k held, cash=$0 → total=$10k.
        - Allocator sees both at 50% weight. B is "held" (no positive entry
          score) so its target is anchored at its current weight, 0.50.
          Contributions for B are empty (the held path).
        - Exit rule sells A entirely → post_sell_cash=$5k.
        - size_buys is then called with positions still pre-sell (A=50, B=50)
          and cash=post_sell_cash=$5k. The caller passes total_value=$10k
          (the same denominator the allocator used) so size_buys' internal
          current_weight for B matches the allocator's view at 0.50.
        - Result: delta(B) = 0.50 - 0.50 = 0, no buy on B. Pre-fix this
          would have computed B's current_weight as 5000/15000 ≈ 0.33,
          fired a phantom buy with empty source.
        """
        sizer = OrderSizer(default_slippage=0.0)
        # Allocator sees A and B both held at 50% weight (no positive
        # contribs for either — they're held, not active).
        allocation = _alloc_result(
            {"A": 0.50, "B": 0.50},
            blended={"A": 0.0, "B": 0.0},
            contribs={"A": {}, "B": {}},
        )
        positions = {"A": 50.0, "B": 50.0}  # pre-sell snapshot
        prices = {"A": 100.0, "B": 100.0}
        post_sell_cash = 5000.0  # A was sold, freeing $5k
        constraints = AllocationConstraints(min_buy_delta=0.02)

        # Pass the pre-sell denominator that the allocator used.
        orders = sizer.size_buys(
            allocation,
            positions,
            prices,
            post_sell_cash,
            constraints,
            total_value=10000.0,
        )
        assert orders == [], "held ticker fired a phantom buy after another ticker sold in the same tick"

    def test_buy_source_is_strongest_contributor(self):
        """Buy attribution picks the largest positive entry-signal contributor."""
        sizer = OrderSizer(default_slippage=0.0)
        allocation = _alloc_result(
            {"A": 0.60},
            blended={"A": 0.5},
            contribs={"A": {"MeanReversion": 0.4, "Momentum": 0.6}},
        )
        positions = {"A": 10.0}
        prices = {"A": 100.0}
        cash = 5000.0
        constraints = AllocationConstraints(min_buy_delta=0.02)
        orders = sizer.size_buys(allocation, positions, prices, cash, constraints, total_value=6000.0)
        assert len(orders) == 1
        assert orders[0].context.source == "Momentum"


class TestSizeExits:
    def test_exit_intent_becomes_sell_order(self):
        """ExitIntent's target_value is converted to a whole-share sell."""
        sizer = OrderSizer(default_slippage=0.0)
        intent = ExitIntent(
            ticker="VOO",
            target_value=500.0,
            source="StopLoss",
            reason="loss exceeded",
        )
        orders = sizer.size_exits([intent], positions={"VOO": 10.0}, prices={"VOO": 100.0})
        assert len(orders) == 1
        assert orders[0].direction == Direction.SELL
        assert orders[0].shares == 5
        assert orders[0].context.source == "StopLoss"
        assert "loss exceeded" in orders[0].context.reason

    def test_exit_capped_by_held_shares(self):
        """Exit can't sell more shares than the position holds."""
        sizer = OrderSizer(default_slippage=0.0)
        intent = ExitIntent(
            ticker="VOO",
            target_value=10000.0,  # asks for 100 shares
            source="StopLoss",
            reason="full liquidation",
        )
        orders = sizer.size_exits([intent], positions={"VOO": 7.0}, prices={"VOO": 100.0})
        assert len(orders) == 1
        assert orders[0].shares == 7

    def test_competing_intents_collapse_to_largest(self):
        """Two rules firing on the same ticker collapse to one sell.

        Regression: ProfitTaking and TrailingStop firing on the same ticker
        in the same tick used to produce two independent sell orders, each
        sized against the full position — selling 2x what was held and
        breaking realized-P&L reconciliation. The intent with the largest
        ``target_value`` should win and be credited with the trade.
        """
        sizer = OrderSizer(default_slippage=0.0)
        intents = [
            ExitIntent(ticker="VOO", target_value=300.0, source="ProfitTaking", reason="up 20%"),
            ExitIntent(ticker="VOO", target_value=1000.0, source="TrailingStop", reason="trail hit"),
        ]
        orders = sizer.size_exits(intents, positions={"VOO": 10.0}, prices={"VOO": 100.0})
        assert len(orders) == 1
        assert orders[0].shares == 10  # full position, not 20
        assert orders[0].context.source == "TrailingStop"

    def test_competing_intents_distinct_tickers_both_fire(self):
        """Collapse is per-ticker — different tickers still produce both sells."""
        sizer = OrderSizer(default_slippage=0.0)
        intents = [
            ExitIntent(ticker="A", target_value=500.0, source="StopLoss", reason="a"),
            ExitIntent(ticker="B", target_value=500.0, source="StopLoss", reason="b"),
        ]
        orders = sizer.size_exits(
            intents,
            positions={"A": 10.0, "B": 10.0},
            prices={"A": 100.0, "B": 100.0},
        )
        assert sorted(o.ticker for o in orders) == ["A", "B"]
