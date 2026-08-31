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
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from midas.data.price_history import PriceHistory
from midas.models import DEFAULT_MAX_POSITION_PCT, DEFAULT_VOL_FLOOR, AllocationConstraints, RiskConfig
from midas.risk import (
    apply_drawdown_overlay,
    inverse_vol_offset,
    predict_portfolio_vol,
    realized_vol,
)
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


def quantile_rank(scores: dict[str, float]) -> dict[str, float]:
    """Rank-transform one rule's positive scores across the cross-section.

    Phase 1.5 of the blend: the i-th smallest positive score
    maps to ``rank / n_positive`` (average rank for ties), so every rule's
    strongest pick scores 1.0 and its weakest positive pick ``1/n``
    regardless of the rule's raw score distribution.

    Zeros stay exactly zero and never join the ranking: the entry-signal
    contract says 0 means "no opinion", and rank-inflating a near-zero
    score to mid-pack would manufacture conviction the rule never
    expressed. Negative scores violate the [0, 1] contract and are
    coerced to 0.0 the same way rather than entering the blend (under
    raw blending they would pull the average down instead).

    Args:
        scores: Ticker -> raw score in [0, 1] for a single rule instance.

    Returns:
        Ticker -> normalized score. NaN inputs (a contract-violating
        live ``score()``) fail both sign comparisons and drop out of the
        result entirely — downstream treats a missing key as abstention,
        which is the safe reading of an undefined score.
    """
    positives = sorted((score, ticker) for ticker, score in scores.items() if score > 0)
    normalized = {ticker: 0.0 for ticker, score in scores.items() if score <= 0}
    total = len(positives)
    index = 0
    while index < total:
        tie_end = index
        while tie_end + 1 < total and positives[tie_end + 1][0] == positives[index][0]:
            tie_end += 1
        # 1-based ranks index+1 .. tie_end+1, averaged across the tie group.
        average_rank = (index + tie_end + 2) / 2
        for _, ticker in positives[index : tie_end + 1]:
            normalized[ticker] = average_rank / total
        index = tie_end + 1
    return normalized


@dataclass
class _ScoredEntry:
    """An entry signal paired with its blending weight."""

    strategy: EntrySignal
    weight: float


@dataclass
class AllocatorRiskTelemetry:
    """Per-allocation snapshot of risk-engine activity.

    Recorded for every ``Allocator.allocate`` call so the engine can build a
    bar-by-bar history. Defaults are inert (``cppi_scale=1.0``,
    ``vol_target_scale=1.0``, etc.) when the corresponding phase is disabled
    or didn't bind, so a disabled risk config produces a flat-line history.
    """

    cppi_scale: float = 1.0
    vol_target_scale: float = 1.0
    vol_target_skipped: bool = False
    vol_target_predicted_vol: float = 0.0
    # Sum of allocator-target weights this bar (the *target* gross, before
    # execution lag, min_buy_delta, exit clamps, etc.). This is the construct-
    # to-budget output and is by design near ``1 - min_cash_pct`` whenever the
    # softmax has any positive scores. *Not* the actual market deployment —
    # that's recorded separately on ``RiskHistory.gross_exposure`` per bar
    # from ``positions_value / total_value`` and is what the ``Avg/Min Gross
    # Exposure`` metrics reflect.
    target_gross_exposure: float = 0.0


@dataclass
class AllocationResult:
    """Target weights plus per-ticker scoring detail from one ``allocate`` call."""

    targets: dict[str, float]
    contributions: dict[str, dict[str, float]]  # ticker -> {strategy_name: score}
    blended_scores: dict[str, float]  # ticker -> blended score
    risk_telemetry: AllocatorRiskTelemetry = field(default_factory=AllocatorRiskTelemetry)


class Allocator:
    """Blends ``EntrySignal`` scores into target portfolio weights via softmax."""

    def __init__(
        self,
        entries: list[tuple[EntrySignal, float]],
        constraints: AllocationConstraints,
        n_tickers: int,
        risk_config: RiskConfig | None = None,
        members: Sequence[Sequence[tuple[EntrySignal, float]]] | None = None,
    ) -> None:
        """Build an allocator from one entry set or K ensemble members.

        ``members`` and ``entries`` are mutually exclusive: ``entries`` is the
        single-member convenience path (today's behavior, bit for bit), and
        ``members=[entries]`` is identical to it by construction.

        Raises:
            ValueError: If both ``entries`` and ``members`` are given.
        """
        if members is not None and entries:
            msg = "pass entries or members, not both (entries is the single-member convenience path)"
            raise ValueError(msg)
        member_lists = [list(entries)] if members is None else [list(member) for member in members]
        if not member_lists:
            member_lists = [list(entries)]
        self._members: list[list[_ScoredEntry]] = [
            [_ScoredEntry(strat, wt) for strat, wt in member] for member in member_lists
        ]
        self._entries: list[_ScoredEntry] = self._members[0]
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
        """All configured entry signals across members, in blending order."""
        return [entry.strategy for member in self._members for entry in member]

    @property
    def risk_config(self) -> RiskConfig | None:
        """The risk-engine configuration, or None when disabled."""
        return self._risk_config

    def precompute_signals(self, price_data: dict[str, PriceHistory]) -> None:
        """Precompute entry-signal scores for all tickers over the full price arrays."""
        self._signal_cache = {}
        for member in self._members:
            self._precompute_member(member, price_data)

    def _precompute_member(self, member: list[_ScoredEntry], price_data: dict[str, PriceHistory]) -> None:
        for entry in member:
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

        telemetry = AllocatorRiskTelemetry()

        # Phase 4a: CPPI drawdown overlay (no-op if not configured).
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
        telemetry.cppi_scale = exposure_scale

        investable = (1.0 - self._constraints.min_cash_pct) * exposure_scale
        base_weight = investable / num_tickers  # fallback when current_weights unknown
        temperature = self._constraints.softmax_temperature

        targets: dict[str, float] = dict.fromkeys(tickers, 0.0)
        contributions, blended_scores, active, held = self._score_tickers(tickers, price_data, ctx)

        offsets: dict[str, float] = {}
        if self._risk_config is not None and self._risk_config.weighting == "inverse_vol" and active:
            offsets, active = self._inverse_vol_offsets(active, price_data, held)

        # Held tickers (Option A) consume their current weight from the
        # investable budget. Remaining budget goes to the active softmax.
        held_total = 0.0
        for ticker in held:
            weight = base_weight if current_weights is None else cur_weights.get(ticker, 0.0)
            targets[ticker] = weight
            held_total += weight

        budget_for_active = max(investable - held_total, 0.0)

        # Phase 2: Softmax construct-to-budget over active tickers.
        self._softmax_allocate(active, blended_scores, budget_for_active, temperature, targets, offsets)

        # Phase 3: Soft position cap. Iteratively clamp any ticker that exceeds
        # max_position_pct and re-softmax the survivors over the reduced
        # remaining budget. Cap *never* forces a sell — it just refuses to
        # allocate more budget. Sells are exclusively ExitRule territory.
        self._apply_cap_with_redistribution(active, blended_scores, budget_for_active, temperature, targets, offsets)

        # Phase 4b: portfolio vol target. Scale all weights down when predicted
        # annualized vol exceeds the target. Skipped if any active ticker has
        # insufficient or degenerate history (constant-price → singular Σ row).
        if self._risk_config is not None and self._risk_config.vol_target is not None and active:
            self._apply_vol_target(active, price_data, targets, telemetry)

        telemetry.target_gross_exposure = float(sum(targets.values()))
        return AllocationResult(targets, contributions, blended_scores, risk_telemetry=telemetry)

    def _score_tickers(
        self,
        tickers: list[str],
        price_data: dict[str, PriceHistory],
        ctx: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, dict[str, float]], dict[str, float], list[str], list[str]]:
        """Score per member, then blend across members (flat mean).

        The single-member path returns its member's result untouched — the
        ensemble-of-one identity is by construction, not by equivalence.
        """
        results = [self._score_member(member, tickers, price_data, ctx) for member in self._members]
        if len(results) == 1:
            return results[0]
        return self._blend_members(tickers, results)

    def _blend_members(
        self,
        tickers: list[str],
        results: list[tuple[dict[str, dict[str, float]], dict[str, float], list[str], list[str]]],
    ) -> tuple[dict[str, dict[str, float]], dict[str, float], list[str], list[str]]:
        """Flat mean of member blends; abstention counts as zero once any member scores.

        A ticker no member scores keeps the hold path. Contributions average
        per strategy name across the members that produced one — display
        attribution only; the blend itself uses member blended scores.
        """
        num_members = len(results)
        contributions: dict[str, dict[str, float]] = {}
        blended: dict[str, float] = {}
        active: list[str] = []
        held: list[str] = []
        for ticker in tickers:
            mean_score = sum(member[1].get(ticker, 0.0) for member in results) / num_members
            per_name: dict[str, list[float]] = {}
            for member in results:
                for name, val in member[0].get(ticker, {}).items():
                    per_name.setdefault(name, []).append(val)
            contributions[ticker] = {name: sum(vals) / len(vals) for name, vals in per_name.items()}
            blended[ticker] = mean_score
            if mean_score > 0:
                active.append(ticker)
            else:
                held.append(ticker)
        return contributions, blended, active, held

    def _score_member(
        self,
        member: list[_ScoredEntry],
        tickers: list[str],
        price_data: dict[str, PriceHistory],
        ctx: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, dict[str, float]], dict[str, float], list[str], list[str]]:
        """Phase 1: score entry signals and compute blended scores per ticker.

        Partitions tickers into `active` (at least one strategy scored > 0) and
        `held` (all strategies abstained or scored 0 — Option A: hold current
        weight). A pure-zero score is treated as "no opinion", consistent with
        the entry-signal contract that 0 means "no contribution to budget".

        Returns:
            ``(contributions, blended_scores, active, held)``.
        """
        contributions: dict[str, dict[str, float]] = {}
        blended_scores: dict[str, float] = {}
        active: list[str] = []
        held: list[str] = []

        # Pass 1: collect raw scores per entry instance across the
        # cross-section. Keyed by entry index, not strategy name — two
        # configs of the same strategy have different score distributions
        # and must be normalized independently.
        per_entry_scores: list[dict[str, float]] = [{} for _ in member]
        scorable: list[str] = []
        for ticker in tickers:
            history = price_data.get(ticker)
            if history is None or len(history) == 0:
                contributions[ticker] = {}
                blended_scores[ticker] = 0.0
                held.append(ticker)
                continue
            scorable.append(ticker)
            ticker_ctx = ctx.get(ticker, {})
            for index, entry in enumerate(member):
                hit, score = self._lookup_score(entry.strategy, ticker, len(history))
                if not hit:
                    score = entry.strategy.score(history, **ticker_ctx)
                if score is not None:
                    per_entry_scores[index][ticker] = score

        # Phase 1.5: per-rule cross-sectional rank transform.
        # A 0.8 from one rule and a 0.8 from another are not the same
        # conviction — raw averaging lets fat-tailed rules dominate the
        # blend. Ranking within each rule's own cross-section makes the
        # rules commensurable before their weights are applied.
        if self._constraints.forecast_scaling == "quantile":
            per_entry_scores = [quantile_rank(scores) for scores in per_entry_scores]

        # Pass 2: blend per ticker. Contributions record the scores that
        # actually entered the blend (normalized when scaling is on), so
        # alert attribution matches what drove the decision.
        for ticker in scorable:
            ticker_contributions: dict[str, float] = {}
            weighted_sum = 0.0
            weight_total = 0.0
            for index, entry in enumerate(member):
                score = per_entry_scores[index].get(ticker)
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

        return contributions, blended_scores, active, held

    def _inverse_vol_offsets(
        self,
        active: list[str],
        price_data: dict[str, PriceHistory],
        held: list[str],
    ) -> tuple[dict[str, float], list[str]]:
        """Phase 2 prep: per-ticker ``-log(vol)`` offsets for inverse-vol weighting.

        Tickers with insufficient or zero vol are reclassified as held
        (Option A) by appending to *held* in place. Offsets are added
        *outside* the /T divider in _softmax_allocate, so inverse-vol
        intensity is invariant to softmax_temperature (an earlier
        implementation used (1/vol)^(1/T) and was rejected for coupling
        the two knobs).

        Returns:
            ``(offsets, still_active)`` — the offset map and the surviving
            active tickers.
        """
        assert self._risk_config is not None
        offsets: dict[str, float] = {}
        still_active: list[str] = []
        for ticker in active:
            history = price_data.get(ticker)
            if history is None or len(history) < self._risk_config.vol_lookback_days + 1:
                held.append(ticker)
                continue
            vol = realized_vol(np.asarray(history.close), self._risk_config.vol_lookback_days)
            offset = inverse_vol_offset(vol, vol_floor=DEFAULT_VOL_FLOOR)
            if math.isnan(offset):
                held.append(ticker)
                continue
            offsets[ticker] = offset
            still_active.append(ticker)
        return offsets, still_active

    def _softmax_allocate(
        self,
        tickers: list[str],
        blended_scores: dict[str, float],
        budget: float,
        temperature: float,
        targets: dict[str, float],
        offsets: dict[str, float] | None = None,
    ) -> None:
        """Distribute `budget` across `tickers` via softmax(blended/T + offset).

        Without offsets, this is the standard softmax: ``exp(score/T)``.
        With offsets (e.g. inverse-vol ``-log(vol)``), each ticker's exponent
        becomes ``score/T + offset``. The offset is *outside* the /T divider
        so its relative impact is invariant to ``temperature``.

        Temperature semantics:
            T → 0   winner-take-all on (score + T * offset) — at very low T
                    the score term dominates the offset.
            T = 1   standard softmax over (score + offset).
            T → ∞   weights ∝ exp(offset), pure offset-driven (e.g. pure
                    inverse-vol regardless of conviction).
        """
        if not tickers or budget <= 0:
            for ticker in tickers:
                targets[ticker] = 0.0
            return
        temp_safe = max(temperature, MIN_TEMPERATURE)
        offsets = offsets or {}
        # Subtract max-exponent for numerical stability — softmax is
        # translation-invariant in the exponent.
        exponents = {ticker: blended_scores[ticker] / temp_safe + offsets.get(ticker, 0.0) for ticker in tickers}
        max_exp = max(exponents.values())
        exps = {ticker: math.exp(exponents[ticker] - max_exp) for ticker in tickers}
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
        offsets: dict[str, float] | None = None,
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
            self._softmax_allocate(survivors, blended_scores, budget, temperature, targets, offsets)

    def _apply_vol_target(
        self,
        active: list[str],
        price_data: dict[str, PriceHistory],
        targets: dict[str, float],
        telemetry: AllocatorRiskTelemetry,
    ) -> None:
        """Scale all active weights so predicted annualized vol ≤ vol_target.

        Skipped silently if any active ticker has insufficient history, a
        non-positive close in the lookback window, or zero realized vol
        (constant-price ticker → singular row in the covariance estimate).
        Reduce-only: when scaling fires, the slack flows to cash; the soft
        cap remains satisfied because scaling can only shrink weights.

        Records ``vol_target_predicted_vol`` and ``vol_target_scale`` on
        ``telemetry``, and flips ``vol_target_skipped=True`` on any silent
        early return.
        """
        assert self._risk_config is not None and self._risk_config.vol_target is not None
        lookback = self._risk_config.vol_lookback_days
        log_returns_per_ticker: list[np.ndarray] = []
        for ticker in active:
            history = price_data.get(ticker)
            if history is None or len(history) < lookback + 1:
                telemetry.vol_target_skipped = True
                return
            window = np.asarray(history.close[-(lookback + 1) :])
            if np.any(window <= 0):
                telemetry.vol_target_skipped = True
                return
            series = np.diff(np.log(window))
            if np.std(series, ddof=1) == 0.0:
                telemetry.vol_target_skipped = True
                return
            log_returns_per_ticker.append(series)
        if not log_returns_per_ticker:
            telemetry.vol_target_skipped = True
            return
        log_returns = np.column_stack(log_returns_per_ticker)
        weights = np.array([targets[ticker] for ticker in active])
        predicted = predict_portfolio_vol(weights, log_returns)
        telemetry.vol_target_predicted_vol = predicted
        target = self._risk_config.vol_target
        if predicted > target > 0.0:
            scale = target / predicted
            telemetry.vol_target_scale = scale
            for ticker in active:
                targets[ticker] *= scale
