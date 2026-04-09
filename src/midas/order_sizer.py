"""OrderSizer: turn allocation decisions into sized Order objects.

Two stateless methods, no shared logic between them:

- ``size_buys`` takes target weights from the Allocator and produces buy
  Orders for any positive delta above ``min_buy_delta``. Never sells —
  drift above target is allowed (winners run; the soft position cap blocks
  further buys but never trims).

- ``size_exits`` takes ``ExitIntent`` objects from ExitRule strategies and
  produces sell Orders, validating that the portfolio actually holds enough
  shares.

Sells and buys come from different sources and have honest, separate
attribution paths. There is no diff-based "rebalance trim" anymore.
"""

from __future__ import annotations

import math

from midas.allocator import AllocationResult
from midas.models import (
    AllocationConstraints,
    Direction,
    ExitIntent,
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
        ``size_exits``.

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
            shares = math.floor(buy_value / slip_price)

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

    def size_exits(
        self,
        intents: list[ExitIntent],
        positions: dict[str, float],
        prices: dict[str, float],
    ) -> list[Order]:
        """Convert exit intents into sized sell orders.

        ``intent.target_value`` is the requested dollar amount to liquidate.
        We size to the largest whole-share count that doesn't exceed both the
        intent and the actual position.

        When multiple exit rules emit intents on the same ticker in the same
        tick (e.g. ProfitTaking *and* TrailingStop both fire), they collapse
        to a single sell: the intent with the largest ``target_value`` wins
        and is credited with the trade. This mirrors the entry side, where
        the allocator's softmax collapses competing buy signals into one
        target weight per ticker — and prevents two rules from each
        liquidating the full position, which would over-sell the ticker and
        break the realized-P&L reconciliation.
        """
        winning: dict[str, ExitIntent] = {}
        for intent in intents:
            current = winning.get(intent.ticker)
            if current is None or intent.target_value > current.target_value:
                winning[intent.ticker] = intent

        orders: list[Order] = []
        for intent in winning.values():
            px = prices.get(intent.ticker, 0.0)
            if px <= 0:
                continue
            if intent.target_value <= 0:
                continue

            slip_price = px * (1 - self._default_slippage)
            target_shares = math.floor(intent.target_value / slip_price)
            held_shares = math.floor(positions.get(intent.ticker, 0.0))
            shares = min(target_shares, held_shares)
            if shares <= 0:
                continue

            orders.append(
                Order(
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
