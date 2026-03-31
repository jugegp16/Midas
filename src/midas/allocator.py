"""Allocator: blends strategy conviction scores into target portfolio weights."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from midas.models import DEFAULT_MAX_POSITION_PCT, AllocationConstraints
from midas.strategies.base import Strategy

log = logging.getLogger(__name__)

# Auto-scaling multipliers for max_position_pct relative to equal weight.
# 2.5x allows meaningful overweighting without extreme concentration.
MAX_POSITION_MULTIPLIER = 2.5
# Warning bounds: below 1.5x, scoring barely moves weights; above 5x, no
# concentration protection.
LOW_POSITION_MULTIPLIER = 1.5
HIGH_POSITION_MULTIPLIER = 5.0


@dataclass
class _ScoredStrategy:
    strategy: Strategy
    weight: float


@dataclass
class _ProtectiveEntry:
    strategy: Strategy
    veto_threshold: float


@dataclass
class AllocationResult:
    targets: dict[str, float]
    contributions: dict[str, dict[str, float]]  # ticker -> {strategy_name: score}
    blended_scores: dict[str, float]  # ticker -> blended score


class Allocator:
    """Blends CONVICTION strategy scores into target portfolio weights.

    Also evaluates PROTECTIVE strategies for vetoes.
    """

    def __init__(
        self,
        conviction_strategies: list[tuple[Strategy, float]],
        protective_strategies: list[tuple[Strategy, float]],
        constraints: AllocationConstraints,
        n_tickers: int,
    ) -> None:
        self._conviction = [_ScoredStrategy(s, w) for s, w in conviction_strategies]
        self._protective = [_ProtectiveEntry(s, vt) for s, vt in protective_strategies]
        self._constraints = constraints
        self._n_tickers = n_tickers

        # Auto-compute max_position_pct if not set
        equal_weight = (1.0 - constraints.min_cash_pct) / max(n_tickers, 1)
        if constraints.max_position_pct is None:
            self._max_position_pct = min(
                DEFAULT_MAX_POSITION_PCT,
                MAX_POSITION_MULTIPLIER * equal_weight,
            )
        else:
            self._max_position_pct = constraints.max_position_pct
            if constraints.max_position_pct < LOW_POSITION_MULTIPLIER * equal_weight:
                log.warning(
                    "max_position_pct (%.2f) is below 1.5x equal weight (%.2f) — scoring may have little effect",
                    constraints.max_position_pct,
                    equal_weight,
                )
            elif constraints.max_position_pct > HIGH_POSITION_MULTIPLIER * equal_weight:
                log.warning(
                    "max_position_pct (%.2f) is above 5x equal weight (%.2f) "
                    "— provides little concentration protection",
                    constraints.max_position_pct,
                    equal_weight,
                )

    def allocate(
        self,
        tickers: list[str],
        price_data: dict[str, np.ndarray],
        context: dict[str, dict[str, object]] | None = None,
    ) -> AllocationResult:
        """Compute target weights for all tickers.

        Args:
            tickers: Active tickers in the portfolio.
            price_data: Ticker -> price series mapping.
            context: Per-ticker context dict (e.g. {"AAPL": {"cost_basis": 150.0}}).

        Returns:
            AllocationResult with target weights, per-ticker contributions,
            and blended scores.
        """
        ctx = context or {}
        n = len(tickers)
        if n == 0:
            return AllocationResult({}, {}, {})

        base_weight = (1.0 - self._constraints.min_cash_pct) / n
        k = self._constraints.sigmoid_steepness

        # Phase 1+2: Score conviction strategies and blend
        contributions: dict[str, dict[str, float]] = {}
        blended_scores: dict[str, float] = {}
        targets: dict[str, float] = {}

        for ticker in tickers:
            prices = price_data.get(ticker)
            if prices is None or len(prices) == 0:
                targets[ticker] = base_weight
                contributions[ticker] = {}
                blended_scores[ticker] = 0.0
                continue

            ticker_ctx = ctx.get(ticker, {})
            ticker_contributions: dict[str, float] = {}
            weighted_sum = 0.0
            weight_total = 0.0

            for entry in self._conviction:
                s = entry.strategy.score(prices, **ticker_ctx)
                if s is not None:
                    ticker_contributions[entry.strategy.name] = s
                    weighted_sum += entry.weight * s
                    weight_total += entry.weight

            contributions[ticker] = ticker_contributions

            blended = weighted_sum / weight_total if weight_total > 0 else 0.0

            blended_scores[ticker] = blended

            # Symmetric sigmoid transform
            sigmoid_val = 1.0 / (1.0 + math.exp(-k * blended))
            targets[ticker] = base_weight * 2.0 * sigmoid_val

        # Phase 3: PROTECTIVE vetoes
        for entry in self._protective:
            for ticker in tickers:
                prices = price_data.get(ticker)
                if prices is None or len(prices) == 0:
                    continue
                ticker_ctx = ctx.get(ticker, {})
                s = entry.strategy.score(prices, **ticker_ctx)
                if s is not None and s <= entry.veto_threshold:
                    targets[ticker] = 0.0
                    # Record protective strategy in contributions
                    contributions.setdefault(ticker, {})[entry.strategy.name] = s

        # Phase 4: Apply constraints
        for ticker in tickers:
            targets[ticker] = max(0.0, min(targets[ticker], self._max_position_pct))

        # Normalize so sum <= 1 - min_cash_pct
        total = sum(targets.values())
        max_total = 1.0 - self._constraints.min_cash_pct
        if total > max_total and total > 0:
            scale = max_total / total
            for ticker in targets:
                targets[ticker] *= scale

        return AllocationResult(targets, contributions, blended_scores)
