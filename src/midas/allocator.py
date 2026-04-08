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
    # ticker -> "cap" when Phase 4 constraints trimmed the target. Used by the
    # Rebalancer to label fallback attribution when no conviction strategy drove
    # the trade.
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

        investable = 1.0 - self._constraints.min_cash_pct
        base_weight = investable / n  # fallback when current_weights unknown
        temperature = self._constraints.softmax_temperature

        contributions: dict[str, dict[str, float]] = {}
        blended_scores: dict[str, float] = {}
        targets: dict[str, float] = dict.fromkeys(tickers, 0.0)
        trim_reasons: dict[str, str] = {}

        # Phase 1: Score conviction strategies and compute blended scores.
        # Partition tickers into `active` (at least one strategy scored) and
        # `held` (all strategies abstained — Option A: hold current weight).
        active: list[str] = []
        held: list[str] = []

        for ticker in tickers:
            prices = price_data.get(ticker)
            if prices is None or len(prices) == 0:
                contributions[ticker] = {}
                blended_scores[ticker] = 0.0
                held.append(ticker)
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
                blended_scores[ticker] = weighted_sum / weight_total
                active.append(ticker)
            else:
                blended_scores[ticker] = 0.0
                held.append(ticker)

        # Phase 2: Protective vetoes. A vetoed ticker is removed from `active`
        # (its softmax slot vanishes and its share of budget flows to peers).
        #
        # KNOWN LIMITATION: the vetoed score is stored in ``contributions`` and
        # the Rebalancer's justification/attribution picker filters by sign
        # (``v < 0`` for SELL). A protective strategy that vetoes with a score
        # >= 0 would therefore have its forced-exit SELL either suppressed by
        # the justification check or mislabeled as ``Rebalancer (unknown)``.
        # All currently shipped protective strategies (StopLoss, TrailingStop)
        # return strictly negative scores when they veto, so this is not
        # exploitable today. The whole protective tier is slated for removal
        # in favor of direct EXIT_RULE order intents — see #26 — at which
        # point this pathway (and the limitation) disappears entirely.
        vetoed: set[str] = set()
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
                    vetoed.add(ticker)
                    contributions.setdefault(ticker, {})[entry.strategy.name] = s
        active = [t for t in active if t not in vetoed]
        held = [t for t in held if t not in vetoed]
        for t in vetoed:
            targets[t] = 0.0

        # Held tickers (Option A) consume their current weight from the
        # investable budget. Remaining budget goes to the active softmax.
        def _held_target(ticker: str) -> float:
            if current_weights is None:
                return base_weight
            return cur_w.get(ticker, 0.0)

        held_total = 0.0
        for t in held:
            w = _held_target(t)
            targets[t] = w
            held_total += w

        budget_for_active = max(investable - held_total, 0.0)

        # Phase 3: Softmax construct-to-budget over active tickers. Sum of
        # softmax weights is exactly budget_for_active by construction, so
        # there is no normalize step.
        self._softmax_allocate(active, blended_scores, budget_for_active, temperature, targets)

        # Phase 4: Position cap with redistribution (Option X). Iteratively
        # clamp any ticker exceeding max_position_pct and re-softmax the
        # survivors over the reduced remaining budget. Usually converges in
        # one pass; bounded by len(active) for safety.
        self._apply_cap_with_redistribution(
            active, blended_scores, budget_for_active, temperature, targets, trim_reasons
        )

        return AllocationResult(targets, contributions, blended_scores, trim_reasons)

    # Temperature floor prevents division-by-zero and bounds exp() growth in the
    # winner-take-all limit. Combined with max-subtraction below, the softmax
    # collapses cleanly to argmax as temperature → 0.
    _MIN_TEMPERATURE = 1e-6

    def _softmax_allocate(
        self,
        tickers: list[str],
        blended_scores: dict[str, float],
        budget: float,
        temperature: float,
        targets: dict[str, float],
    ) -> None:
        """Distribute `budget` across `tickers` via softmax(blended_scores / T).

        Temperature semantics:
            T → 0   winner-take-all (budget to the argmax ticker)
            T = 1   standard softmax over raw scores
            T → ∞   uniform split regardless of conviction
        """
        if not tickers or budget <= 0:
            for t in tickers:
                targets[t] = 0.0
            return
        t_safe = max(temperature, self._MIN_TEMPERATURE)
        # Subtract max for numerical stability — softmax is translation-invariant.
        max_score = max(blended_scores[t] for t in tickers)
        exps = {t: math.exp((blended_scores[t] - max_score) / t_safe) for t in tickers}
        z = sum(exps.values())
        if z <= 0:
            # Degenerate (shouldn't happen after max-subtraction, but be safe).
            equal = budget / len(tickers)
            for t in tickers:
                targets[t] = equal
            return
        for t in tickers:
            targets[t] = budget * exps[t] / z

    def _apply_cap_with_redistribution(
        self,
        active: list[str],
        blended_scores: dict[str, float],
        initial_budget: float,
        temperature: float,
        targets: dict[str, float],
        trim_reasons: dict[str, str],
    ) -> None:
        """Clamp targets exceeding max_position_pct; re-softmax the survivors.

        Option X: when a cap trims a ticker, the freed budget is redistributed
        to uncapped tickers in proportion to their softmax weight (achieved by
        re-running softmax over the survivors with the reduced budget).
        """
        cap = self._max_position_pct
        survivors = list(active)
        budget = initial_budget
        # At most one iteration per ticker (each pass pins at least one).
        for _ in range(len(active) + 1):
            over = [t for t in survivors if targets[t] > cap + 1e-12]
            if not over:
                return
            for t in over:
                targets[t] = cap
                trim_reasons[t] = "cap"
                budget -= cap
                survivors.remove(t)
            if not survivors or budget <= 0:
                for t in survivors:
                    targets[t] = 0.0
                return
            self._softmax_allocate(survivors, blended_scores, budget, temperature, targets)
