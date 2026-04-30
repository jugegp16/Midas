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


@dataclass
class _ScoredEntry:
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

        # Phase 2 prep: when ``weighting=inverse_vol``, compute per-ticker
        # ``-log(vol)`` offsets and reclassify tickers with insufficient or
        # zero vol as held (Option A). Offsets are added *outside* the /T
        # divider in _softmax_allocate, so inverse-vol intensity is invariant
        # to softmax_temperature (PR #63 used (1/vol)^(1/T) and was rejected).
        offsets: dict[str, float] = {}
        if self._risk_config is not None and self._risk_config.weighting == "inverse_vol" and active:
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
            active = still_active

        held_total = 0.0
        for ticker in held:
            weight = _held_target(ticker)
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
