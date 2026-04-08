"""Allocator: blends strategy conviction scores into target portfolio weights."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

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
    # ticker -> "cap" | "normalize" when Phase 4 constraints trimmed the target.
    # Used by the Rebalancer to label fallback attribution when no conviction
    # strategy drove the trade.
    trim_reasons: dict[str, str]


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
        self._conviction: list[_ScoredStrategy] = [_ScoredStrategy(s, w) for s, w in conviction_strategies]
        self._protective: list[_ProtectiveEntry] = [_ProtectiveEntry(s, vt) for s, vt in protective_strategies]
        self._constraints = constraints
        self._n_tickers = n_tickers
        self._signal_cache: dict[int, dict[str, np.ndarray]] = {}

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

    @property
    def strategies(self) -> list[Strategy]:
        """All strategies the allocator owns (conviction + protective)."""
        return [s.strategy for s in self._conviction] + [e.strategy for e in self._protective]

    def precompute_signals(self, price_data: dict[str, np.ndarray]) -> None:
        """Precompute strategy scores for all tickers over the full price arrays.

        Call once before the simulation loop.  During ``allocate()``, cached
        scores are looked up by array index instead of calling ``score()``.
        """
        self._signal_cache = {}
        strategies = [s.strategy for s in self._conviction] + [e.strategy for e in self._protective]
        for strat in strategies:
            cache: dict[str, np.ndarray] = {}
            for ticker, prices in price_data.items():
                result = strat.precompute(prices)
                if result is not None:
                    cache[ticker] = result
            if cache:
                self._signal_cache[id(strat)] = cache

    def _lookup_score(self, strategy: Strategy, ticker: str, prices_len: int) -> tuple[bool, float | None]:
        """Look up a precomputed score.  Returns (hit, score)."""
        strat_cache = self._signal_cache.get(id(strategy))
        if strat_cache is not None and ticker in strat_cache:
            val = strat_cache[ticker][prices_len - 1]
            return True, (None if np.isnan(val) else float(val))
        return False, None

    def allocate(
        self,
        tickers: list[str],
        price_data: dict[str, np.ndarray],
        context: dict[str, dict[str, Any]] | None = None,
        current_weights: dict[str, float] | None = None,
    ) -> AllocationResult:
        """Compute target weights for all tickers.

        Args:
            tickers: Active tickers in the portfolio.
            price_data: Ticker -> price series mapping.
            context: Per-ticker context dict (e.g. {"AAPL": {"cost_basis": 150.0}}).
            current_weights: Ticker -> current portfolio weight. When no
                conviction strategy scores a ticker, the allocator holds the
                current weight instead of reverting to equal-weight (Option A:
                neutral = hold). If omitted, falls back to equal-weight base.

        Returns:
            AllocationResult with target weights, per-ticker contributions,
            blended scores, and trim reasons.
        """
        ctx = context or {}
        cur_w = current_weights or {}
        n = len(tickers)
        if n == 0:
            return AllocationResult({}, {}, {}, {})

        base_weight = (1.0 - self._constraints.min_cash_pct) / n
        k = self._constraints.sigmoid_steepness

        # Phase 1+2: Score conviction strategies and blend
        contributions: dict[str, dict[str, float]] = {}
        blended_scores: dict[str, float] = {}
        targets: dict[str, float] = {}
        trim_reasons: dict[str, str] = {}

        def _neutral_target(ticker: str) -> float:
            # Neutral = hold (Option A). Falls back to base_weight only when
            # no current weight is known (e.g. first-ever allocation).
            if current_weights is None:
                return base_weight
            return cur_w.get(ticker, 0.0)

        for ticker in tickers:
            prices = price_data.get(ticker)
            if prices is None or len(prices) == 0:
                targets[ticker] = _neutral_target(ticker)
                contributions[ticker] = {}
                blended_scores[ticker] = 0.0
                continue

            ticker_ctx = ctx.get(ticker, {})
            ticker_contributions: dict[str, float] = {}
            weighted_sum = 0.0
            weight_total = 0.0

            for scored in self._conviction:
                hit, s = self._lookup_score(scored.strategy, ticker, len(prices))
                if not hit:
                    s = scored.strategy.score(prices, **ticker_ctx)
                if s is not None:
                    ticker_contributions[scored.strategy.name] = s
                    weighted_sum += scored.weight * s
                    weight_total += scored.weight

            contributions[ticker] = ticker_contributions

            if weight_total > 0:
                blended = weighted_sum / weight_total
                blended_scores[ticker] = blended
                # Symmetric sigmoid transform
                sigmoid_val = 1.0 / (1.0 + math.exp(-k * blended))
                targets[ticker] = base_weight * 2.0 * sigmoid_val
            else:
                # No conviction strategy scored — hold current weight rather
                # than drifting to base_weight and forcing drift-correction
                # trades on every rebalance.
                blended_scores[ticker] = 0.0
                targets[ticker] = _neutral_target(ticker)

        # Phase 3: PROTECTIVE vetoes
        for entry in self._protective:
            for ticker in tickers:
                prices = price_data.get(ticker)
                if prices is None or len(prices) == 0:
                    continue
                hit, s = self._lookup_score(entry.strategy, ticker, len(prices))
                if not hit:
                    ticker_ctx = ctx.get(ticker, {})
                    s = entry.strategy.score(prices, **ticker_ctx)
                if s is not None and s <= entry.veto_threshold:
                    targets[ticker] = 0.0
                    # Record protective strategy in contributions
                    contributions.setdefault(ticker, {})[entry.strategy.name] = s

        # Phase 4: Apply constraints
        for ticker in tickers:
            pre = targets[ticker]
            clamped = max(0.0, min(pre, self._max_position_pct))
            if clamped < pre:
                trim_reasons[ticker] = "cap"
            targets[ticker] = clamped

        # Normalize so sum <= 1 - min_cash_pct
        total = sum(targets.values())
        max_total = 1.0 - self._constraints.min_cash_pct
        if total > max_total and total > 0:
            scale = max_total / total
            for ticker in targets:
                targets[ticker] *= scale
                # "cap" takes precedence — it's more specific than normalize.
                trim_reasons.setdefault(ticker, "normalize")

        return AllocationResult(targets, contributions, blended_scores, trim_reasons)
