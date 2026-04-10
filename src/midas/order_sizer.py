"""OrderSizer: turn allocation decisions into sized Order objects.

Two stateless methods:

- ``size_buys`` takes target weights from the Allocator and produces buy
  Orders for any positive delta above ``min_buy_delta``. Never sells —
  drift above target is allowed (winners run; the soft position cap blocks
  further buys but never trims).

- ``size_sells`` takes clamped target weights (after exit rules have
  reduced them) and produces sell Orders for any negative delta where
  the current weight exceeds the clamped target.

Buys come from entry-signal-driven targets; sells come from exit-rule
clamping. Attribution is honest and separate.
"""

from __future__ import annotations

import math

from midas.allocator import AllocationResult
from midas.models import (
    AllocationConstraints,
    Direction,
    Order,
    OrderContext,
)

DEFAULT_SLIPPAGE = 0.0005  # 0.05%
DEFAULT_CIRCUIT_BREAKER_PCT = 0.25  # max 25% of portfolio value per day


class OrderSizer:
    """Sizes allocator targets and exit intents into broker-ready orders."""

    def __init__(
        self,
        default_slippage: float = DEFAULT_SLIPPAGE,
        circuit_breaker_pct: float = DEFAULT_CIRCUIT_BREAKER_PCT,
    ) -> None:
        self._default_slippage = default_slippage
        self._circuit_breaker_pct = circuit_breaker_pct

    def size_buys(
        self,
        allocation: AllocationResult,
        positions: dict[str, float],
        prices: dict[str, float],
        cash: float,
        constraints: AllocationConstraints,
        total_value: float,
    ) -> list[Order]:
        """Size buy orders from positive target-vs-current deltas.

        Sells are not produced here — drift above target is allowed and the
        soft cap only blocks further buys. Any required sells come from
        ``size_sells``.

        ``total_value`` must be the same portfolio basis the allocator used to
        compute ``allocation.targets``. The caller is responsible for passing a
        consistent value so that the per-ticker current weights computed inside
        ``size_buys`` line up with the allocator's view; otherwise held tickers
        can fire phantom buys when sells in the same tick free cash and shift
        the denominator.
        """
        if total_value <= 0:
            return []

        # Compute current weights for delta calculations using the allocator's
        # basis (total_value), not post-sell cash.
        current_weights: dict[str, float] = {}
        for ticker in allocation.targets:
            pos = positions.get(ticker, 0.0)
            px = prices.get(ticker, 0.0)
            current_weights[ticker] = (pos * px) / total_value

        # Circuit breaker: cap daily deployment.
        max_deploy = total_value * self._circuit_breaker_pct
        deployed = 0.0
        available = cash

        buys: list[Order] = []
        for ticker, target_w in allocation.targets.items():
            current_w = current_weights.get(ticker, 0.0)
            delta = target_w - current_w

            # Buy-only: only positive deltas above the threshold.
            if delta < constraints.min_buy_delta:
                continue

            px = prices.get(ticker, 0.0)
            if px <= 0:
                continue

            buy_value = delta * total_value
            slip_price = px * (1 + self._default_slippage)
            shares = math.floor(buy_value / px)

            # Cash constraint.
            affordable = math.floor(available / slip_price)
            shares = min(shares, affordable)

            # Circuit breaker.
            cb_shares = math.floor((max_deploy - deployed) / slip_price)
            shares = min(shares, cb_shares)

            if shares <= 0:
                continue

            cost = shares * slip_price
            available -= cost
            deployed += cost

            buys.append(
                Order(
                    ticker=ticker,
                    direction=Direction.BUY,
                    shares=shares,
                    price=round(slip_price, 4),
                    estimated_value=round(cost, 2),
                    context=self._build_buy_context(ticker, allocation, target_w, current_w),
                )
            )

        return buys

    def size_sells(
        self,
        clamped_targets: dict[str, float],
        positions: dict[str, float],
        prices: dict[str, float],
        total_value: float,
        clamp_attribution: dict[str, tuple[str, str]],
    ) -> list[Order]:
        """Sell orders from negative deltas (clamped target < current weight).

        ``clamped_targets`` maps ticker → clamped target weight (after exit
        rules have reduced the allocator's proposed targets).

        ``clamp_attribution`` maps ticker → (source, reason) for attribution
        when an exit rule fired. Only tickers present in this dict will
        generate sell orders.

        Sells are sized from the negative delta between clamped target and
        current weight, capped at held shares. Slippage is applied
        conservatively (sell price below market).
        """
        if total_value <= 0:
            return []

        orders: list[Order] = []
        for ticker, clamped_w in clamped_targets.items():
            if ticker not in clamp_attribution:
                continue

            pos = positions.get(ticker, 0.0)
            px = prices.get(ticker, 0.0)
            if pos <= 0 or px <= 0:
                continue

            current_w = (pos * px) / total_value
            delta = current_w - clamped_w
            if delta <= 0:
                continue

            sell_value = delta * total_value
            slip_price = px * (1 - self._default_slippage)
            shares = math.floor(sell_value / px)
            shares = min(shares, math.floor(pos))
            if shares <= 0:
                continue

            source, reason = clamp_attribution[ticker]
            orders.append(
                Order(
                    ticker=ticker,
                    direction=Direction.SELL,
                    shares=shares,
                    price=round(slip_price, 4),
                    estimated_value=round(shares * slip_price, 2),
                    context=OrderContext(
                        contributions={},
                        blended_score=0.0,
                        target_weight=clamped_w,
                        current_weight=round(current_w, 6),
                        reason=reason,
                        source=source,
                    ),
                )
            )
        return orders

    @staticmethod
    def _build_buy_context(
        ticker: str,
        allocation: AllocationResult,
        target_weight: float,
        current_weight: float,
    ) -> OrderContext:
        """Attribute a buy to its strongest contributing entry signal."""
        contribs = allocation.contributions.get(ticker, {})
        blended = allocation.blended_scores.get(ticker, 0.0)
        reason = (
            f"Buy {ticker}: target {target_weight:.1%} vs current {current_weight:.1%} (blended score {blended:+.3f})"
        )
        # Buys are always driven by entry signals — pick the largest positive
        # contributor. Entry signals score in [0, 1], so a non-empty
        # contributions dict guarantees at least one positive entry when this
        # buy actually fires.
        positive = {k: v for k, v in contribs.items() if v > 0}
        source = max(positive, key=lambda k: positive[k]) if positive else ""
        return OrderContext(
            contributions=contribs,
            blended_score=blended,
            target_weight=target_weight,
            current_weight=current_weight,
            reason=reason,
            source=source,
        )
