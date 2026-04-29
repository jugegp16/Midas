"""Allocator: blends entry signal scores into target portfolio weights.

Pure entry-side concern. Exits live in ``ExitRule`` strategies and never
participate in blending. The allocator's only job is to turn N entry signal
scores into N target weights under the configured global constraints
(``min_cash_pct``, ``max_position_pct``, ``softmax_temperature``).

Soft cap semantics: ``max_position_pct`` is enforced by *clamping* a
ticker's softmax weight at the cap and giving the clamped survivors the
freed budget on the next pass. The cap never produces a sell — exits are
exclusively the responsibility of ExitRule strategies.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from midas.data.price_history import PriceHistory
from midas.models import DEFAULT_MAX_POSITION_PCT, AllocationConstraints, RiskConfig
from midas.risk import apply_drawdown_overlay
from midas.strategies.base import EntrySignal

log = logging.getLogger(__name__)

# Auto-scaling multipliers for max_position_pct relative to equal weight.
# 2.5x allows meaningful overweighting without extreme concentration.
MAX_POSITION_MULTIPLIER = 2.5
# Warning bounds: below 1.5x, scoring barely moves weights; above 5x, no
# concentration protection.
LOW_POSITION_MULTIPLIER = 1.5
HIGH_POSITION_MULTIPLIER = 5.0

# Temperature floor prevents division-by-zero and bounds exp() growth in the
# winner-take-all limit. Combined with max-subtraction in _softmax_allocate,
# the softmax collapses cleanly to argmax as temperature → 0.
MIN_TEMPERATURE = 1e-6


@dataclass
class _ScoredEntry:
    strategy: EntrySignal
    weight: float


@dataclass
class AllocationResult:
    targets: dict[str, float]
    contributions: dict[str, dict[str, float]]  # ticker -> {strategy_name: score}
    blended_scores: dict[str, float]  # ticker -> blended score


class Allocator:
    """Blends ``EntrySignal`` scores into target portfolio weights via softmax."""

    def __init__(
        self,
        entries: list[tuple[EntrySignal, float]],
        constraints: AllocationConstraints,
        n_tickers: int,
        risk_config: RiskConfig | None = None,
    ) -> None:
        self._entries: list[_ScoredEntry] = [_ScoredEntry(strat, wt) for strat, wt in entries]
        self._constraints = constraints
        self._n_tickers = n_tickers
        self._risk_config = risk_config
        self._signal_cache: dict[int, dict[str, np.ndarray]] = {}

        # Auto-compute max_position_pct if not set.
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
    def strategies(self) -> list[EntrySignal]:
        return [entry.strategy for entry in self._entries]

    @property
    def risk_config(self) -> RiskConfig | None:
        return self._risk_config

    def precompute_signals(self, price_data: dict[str, PriceHistory]) -> None:
        """Precompute entry-signal scores for all tickers over the full price arrays."""
        self._signal_cache = {}
        for entry in self._entries:
            cache: dict[str, np.ndarray] = {}
            for ticker, history in price_data.items():
                result = entry.strategy.precompute(history)
                if result is not None:
                    cache[ticker] = result
            if cache:
                self._signal_cache[id(entry.strategy)] = cache

    def _lookup_score(self, strategy: EntrySignal, ticker: str, history_len: int) -> tuple[bool, float | None]:
        """Look up a precomputed score.  Returns (hit, score)."""
        strat_cache = self._signal_cache.get(id(strategy))
        if strat_cache is not None and ticker in strat_cache:
            val = strat_cache[ticker][history_len - 1]
            return True, (None if np.isnan(val) else float(val))
        return False, None

    def allocate(
        self,
        tickers: list[str],
        price_data: dict[str, PriceHistory],
        context: dict[str, dict[str, Any]] | None = None,
        current_weights: dict[str, float] | None = None,
        current_drawdown: float = 0.0,
    ) -> AllocationResult:
        """Compute target weights for all tickers.

        Args:
            tickers: Active tickers in the portfolio.
            price_data: Ticker -> price series mapping.
            context: Per-ticker context dict (e.g. {"AAPL": {"cost_basis": 150.0}}).
            current_weights: Ticker -> current portfolio weight. When no entry
                signal scores a ticker, the allocator holds the current weight
                instead of reverting to equal-weight (Option A: neutral = hold).
                If omitted, falls back to equal-weight base.
            current_drawdown: Positive fraction (e.g. ``0.20`` for 20% drawdown
                from running peak), used by the optional CPPI overlay. Default
                ``0.0`` is a no-op when the overlay is disabled.

        Returns:
            AllocationResult with target weights, per-ticker contributions,
            and blended scores.
        """
        ctx = context or {}
        cur_weights = current_weights or {}
        num_tickers = len(tickers)
        if num_tickers == 0:
            return AllocationResult({}, {}, {})

        # Phase 0: CPPI drawdown overlay (no-op if not configured).
        exposure_scale = 1.0
        if (
            self._risk_config is not None
            and self._risk_config.drawdown_penalty is not None
            and self._risk_config.drawdown_floor is not None
        ):
            exposure_scale = apply_drawdown_overlay(
                current_drawdown=current_drawdown,
                penalty=self._risk_config.drawdown_penalty,
                floor=self._risk_config.drawdown_floor,
            )

        investable = (1.0 - self._constraints.min_cash_pct) * exposure_scale
        base_weight = investable / num_tickers  # fallback when current_weights unknown
        temperature = self._constraints.softmax_temperature

        contributions: dict[str, dict[str, float]] = {}
        blended_scores: dict[str, float] = {}
        targets: dict[str, float] = dict.fromkeys(tickers, 0.0)

        # Phase 1: Score entry signals and compute blended scores.
        # Partition tickers into `active` (at least one strategy scored > 0) and
        # `held` (all strategies abstained or scored 0 — Option A: hold current
        # weight). A pure-zero score is treated as "no opinion", consistent with
        # the entry-signal contract that 0 means "no contribution to budget".
        active: list[str] = []
        held: list[str] = []

        for ticker in tickers:
            history = price_data.get(ticker)
            if history is None or len(history) == 0:
                contributions[ticker] = {}
                blended_scores[ticker] = 0.0
                held.append(ticker)
                continue

            ticker_ctx = ctx.get(ticker, {})
            ticker_contributions: dict[str, float] = {}
            weighted_sum = 0.0
            weight_total = 0.0

            for entry in self._entries:
                hit, score = self._lookup_score(entry.strategy, ticker, len(history))
                if not hit:
                    score = entry.strategy.score(history, **ticker_ctx)
                if score is not None:
                    ticker_contributions[entry.strategy.name] = score
                    weighted_sum += entry.weight * score
                    weight_total += entry.weight

            contributions[ticker] = ticker_contributions

            if weight_total > 0 and weighted_sum > 0:
                blended_scores[ticker] = weighted_sum / weight_total
                active.append(ticker)
            else:
                blended_scores[ticker] = 0.0
                held.append(ticker)

        # Held tickers (Option A) consume their current weight from the
        # investable budget. Remaining budget goes to the active softmax.
        def _held_target(ticker: str) -> float:
            if current_weights is None:
                return base_weight
            return cur_weights.get(ticker, 0.0)

        held_total = 0.0
        for ticker in held:
            weight = _held_target(ticker)
            targets[ticker] = weight
            held_total += weight

        budget_for_active = max(investable - held_total, 0.0)

        # Phase 2: Softmax construct-to-budget over active tickers.
        self._softmax_allocate(active, blended_scores, budget_for_active, temperature, targets)

        # Phase 3: Soft position cap. Iteratively clamp any ticker that exceeds
        # max_position_pct and re-softmax the survivors over the reduced
        # remaining budget. Cap *never* forces a sell — it just refuses to
        # allocate more budget. Sells are exclusively ExitRule territory.
        self._apply_cap_with_redistribution(active, blended_scores, budget_for_active, temperature, targets)

        return AllocationResult(targets, contributions, blended_scores)

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
            for ticker in tickers:
                targets[ticker] = 0.0
            return
        temp_safe = max(temperature, MIN_TEMPERATURE)
        # Subtract max for numerical stability — softmax is translation-invariant.
        max_score = max(blended_scores[ticker] for ticker in tickers)
        exps = {ticker: math.exp((blended_scores[ticker] - max_score) / temp_safe) for ticker in tickers}
        total_exp = sum(exps.values())
        for ticker in tickers:
            targets[ticker] = budget * exps[ticker] / total_exp

    def _apply_cap_with_redistribution(
        self,
        active: list[str],
        blended_scores: dict[str, float],
        initial_budget: float,
        temperature: float,
        targets: dict[str, float],
    ) -> None:
        """Clamp targets exceeding max_position_pct; re-softmax the survivors.

        Soft cap: when a ticker hits the cap, the freed budget is redistributed
        to uncapped tickers in proportion to their softmax weight. The cap
        never forces a sell — it can only refuse to allocate more buy budget.
        """
        cap = self._max_position_pct
        survivors = list(active)
        budget = initial_budget
        # At most one iteration per ticker (each pass pins at least one).
        for _ in range(len(active) + 1):
            over = [ticker for ticker in survivors if targets[ticker] > cap + 1e-12]
            if not over:
                return
            for ticker in over:
                targets[ticker] = cap
                budget -= cap
                survivors.remove(ticker)
            if not survivors or budget <= 0:
                # Caps consumed the entire investable budget. Survivors get
                # zero new allocation; they'll naturally drift until an
                # ExitRule fires.
                for ticker in survivors:
                    targets[ticker] = 0.0
                return
            self._softmax_allocate(survivors, blended_scores, budget, temperature, targets)
