"""Rebalancer: diffs target weights vs current holdings to generate orders."""

from __future__ import annotations

import math
from dataclasses import dataclass

from midas.allocator import AllocationResult
from midas.models import (
    AllocationConstraints,
    Direction,
    MechanicalIntent,
    Order,
    OrderContext,
)

DEFAULT_SLIPPAGE = 0.0005  # 0.05%
DEFAULT_CIRCUIT_BREAKER_PCT = 0.25  # max 25% of portfolio value per day


@dataclass
class RebalancerConfig:
    default_slippage: float = DEFAULT_SLIPPAGE
    circuit_breaker_pct: float = DEFAULT_CIRCUIT_BREAKER_PCT


class Rebalancer:
    """Generates orders by diffing target weights against current positions."""

    def __init__(self, config: RebalancerConfig | None = None) -> None:
        self._config = config or RebalancerConfig()

    def generate_orders(
        self,
        allocation: AllocationResult,
        positions: dict[str, float],
        prices: dict[str, float],
        cash: float,
        constraints: AllocationConstraints,
    ) -> list[Order]:
        """Generate rebalance orders (sells first, then buys).

        Args:
            allocation: Result from Allocator.allocate().
            positions: Ticker -> share count.
            prices: Ticker -> current price.
            cash: Available cash before rebalancing.
            constraints: Allocation constraints (for rebalance_threshold).

        Returns:
            List of orders, sells before buys.
        """
        total_value = cash + sum(
            positions.get(t, 0.0) * prices.get(t, 0.0)
            for t in allocation.targets
        )
        if total_value <= 0:
            return []

        # Compute current weights
        current_weights: dict[str, float] = {}
        for ticker in allocation.targets:
            pos = positions.get(ticker, 0.0)
            px = prices.get(ticker, 0.0)
            current_weights[ticker] = (
                (pos * px) / total_value if total_value > 0 else 0.0
            )

        # Compute deltas, skip below threshold
        sells: list[Order] = []
        buys: list[tuple[str, float, float]] = []  # (ticker, delta, target_weight)

        for ticker, target_w in allocation.targets.items():
            current_w = current_weights.get(ticker, 0.0)
            delta = target_w - current_w

            if abs(delta) < constraints.rebalance_threshold:
                continue

            px = prices.get(ticker, 0.0)
            if px <= 0:
                continue

            if delta < 0:
                # Sell
                sell_value = abs(delta) * total_value
                slip_price = px * (1 - self._config.default_slippage)
                shares = math.floor(sell_value / slip_price)
                max_shares = math.floor(positions.get(ticker, 0.0))
                shares = min(shares, max_shares)
                if shares <= 0:
                    continue
                order = Order(
                    ticker=ticker,
                    direction=Direction.SELL,
                    shares=shares,
                    price=round(slip_price, 4),
                    estimated_value=round(shares * slip_price, 2),
                    context=self._build_context(
                        ticker, allocation, target_w, current_w, Direction.SELL
                    ),
                )
                sells.append(order)
            else:
                buys.append((ticker, delta, target_w))

        # Cash freed by sells
        freed_cash = sum(o.estimated_value for o in sells)
        available = cash + freed_cash

        # Circuit breaker: cap daily deployment
        max_deploy = total_value * self._config.circuit_breaker_pct
        deployed = 0.0

        buy_orders: list[Order] = []
        for ticker, delta, target_w in buys:
            px = prices.get(ticker, 0.0)
            if px <= 0:
                continue

            buy_value = delta * total_value
            slip_price = px * (1 + self._config.default_slippage)
            shares = math.floor(buy_value / slip_price)

            # Cash constraint
            affordable = math.floor(available / slip_price)
            shares = min(shares, affordable)

            # Circuit breaker
            cb_shares = math.floor((max_deploy - deployed) / slip_price)
            shares = min(shares, cb_shares)

            if shares <= 0:
                continue

            cost = shares * slip_price
            available -= cost
            deployed += cost
            current_w = current_weights.get(ticker, 0.0)

            order = Order(
                ticker=ticker,
                direction=Direction.BUY,
                shares=shares,
                price=round(slip_price, 4),
                estimated_value=round(cost, 2),
                context=self._build_context(
                    ticker, allocation, target_w, current_w, Direction.BUY
                ),
            )
            buy_orders.append(order)

        return sells + buy_orders

    def size_mechanical(
        self,
        intents: list[MechanicalIntent],
        cash: float,
        prices: dict[str, float],
    ) -> list[Order]:
        """Convert mechanical intents into sized orders.

        Args:
            intents: From mechanical strategies (e.g. DCA).
            cash: Available cash after rebalance orders.
            prices: Ticker -> current price.

        Returns:
            List of sized orders.
        """
        orders: list[Order] = []
        available = cash

        for intent in intents:
            px = prices.get(intent.ticker, 0.0)
            if px <= 0:
                continue

            if intent.direction == Direction.BUY:
                slip_price = px * (1 + self._config.default_slippage)
                target_shares = math.floor(intent.target_value / slip_price)
                affordable = math.floor(available / slip_price)
                shares = min(target_shares, affordable)
                if shares <= 0:
                    continue
                cost = shares * slip_price
                available -= cost
                orders.append(Order(
                    ticker=intent.ticker,
                    direction=Direction.BUY,
                    shares=shares,
                    price=round(slip_price, 4),
                    estimated_value=round(cost, 2),
                    context=OrderContext(
                        contributions={},
                        blended_score=0.0,
                        target_weight=0.0,
                        current_weight=0.0,
                        reason=intent.reason,
                        source=intent.source,
                    ),
                ))
            else:
                # Mechanical sells are unusual but supported
                slip_price = px * (1 - self._config.default_slippage)
                shares = math.floor(intent.target_value / slip_price)
                if shares <= 0:
                    continue
                orders.append(Order(
                    ticker=intent.ticker,
                    direction=Direction.SELL,
                    shares=shares,
                    price=round(slip_price, 4),
                    estimated_value=round(shares * slip_price, 2),
                    context=OrderContext(
                        contributions={},
                        blended_score=0.0,
                        target_weight=0.0,
                        current_weight=0.0,
                        reason=intent.reason,
                        source=intent.source,
                    ),
                ))

        return orders

    @staticmethod
    def _build_context(
        ticker: str,
        allocation: AllocationResult,
        target_weight: float,
        current_weight: float,
        direction: Direction,
    ) -> OrderContext:
        contribs = allocation.contributions.get(ticker, {})
        blended = allocation.blended_scores.get(ticker, 0.0)
        action = "Buy" if direction == Direction.BUY else "Sell"
        reason = (
            f"{action} {ticker}: target {target_weight:.1%} vs "
            f"current {current_weight:.1%} (blended score {blended:+.3f})"
        )
        # Primary strategy = highest absolute contribution
        source = (
            max(contribs, key=lambda k: abs(contribs[k]))
            if contribs else "Rebalancer"
        )
        return OrderContext(
            contributions=contribs,
            blended_score=blended,
            target_weight=target_weight,
            current_weight=current_weight,
            reason=reason,
            source=source,
        )
