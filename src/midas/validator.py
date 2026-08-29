"""Policy validation: execute the deployment policy exactly as live will.

Cadence-driven folds, one fit per fold via ``fit_as_of``, degradation-gated
adoption evaluated over full daily paths, and a portfolio that carries
across fold boundaries — emitting the baselines artifact the monitor and
sweep consume.
"""

from __future__ import annotations

from collections.abc import Sequence

from midas.baselines import FoldRecord
from midas.policy import AdoptTrigger

# Arming thresholds: an arm stays silent until enough validation history
# exists to give it meaning — mirroring what live could have known at the
# same point in time.
DRAWDOWN_ARM_MIN_FOLDS = 1
PERCENTILE_ARM_MIN_FOLDS = 5


def trigger_fired(
    prior_folds: Sequence[FoldRecord],
    current_daily_returns: Sequence[float],
    trigger: AdoptTrigger,
) -> bool:
    """Whether the degradation trigger fired at any point along *current* path.

    Latched semantics: the answer is the max over the path's day offsets,
    because live evaluates nightly and a mid-window breach counts even if
    the window later recovers.

    Arms:
        drawdown (>= 1 prior fold): the path's drawdown from its window
            peak exceeds the worst prior-fold drawdown.
        percentile (>= 5 prior folds): the cumulative return at some day
            offset sits below the configured percentile of prior folds'
            cumulative returns at the same offset (shorter folds skipped).
    """
    if not current_daily_returns:
        return False

    if trigger.drawdown_breach and len(prior_folds) >= DRAWDOWN_ARM_MIN_FOLDS:
        worst_prior = max(fold.drawdown for fold in prior_folds)
        if _max_drawdown_along(current_daily_returns) > worst_prior:
            return True

    if len(prior_folds) >= PERCENTILE_ARM_MIN_FOLDS:
        prior_cumulative = [_cumulative(fold.daily_returns) for fold in prior_folds]
        current_cumulative = _cumulative(current_daily_returns)
        for offset, value in enumerate(current_cumulative):
            at_offset = sorted(path[offset] for path in prior_cumulative if len(path) > offset)
            if len(at_offset) < PERCENTILE_ARM_MIN_FOLDS:
                continue
            cutoff_index = int(len(at_offset) * trigger.oos_percentile_below / 100.0)
            cutoff = at_offset[min(cutoff_index, len(at_offset) - 1)]
            if value < cutoff:
                return True

    return False


def _cumulative(daily_returns: Sequence[float]) -> list[float]:
    values: list[float] = []
    value = 1.0
    for ret in daily_returns:
        value *= 1.0 + ret
        values.append(value - 1.0)
    return values


def _max_drawdown_along(daily_returns: Sequence[float]) -> float:
    value = 1.0
    peak = 1.0
    worst = 0.0
    for ret in daily_returns:
        value *= 1.0 + ret
        peak = max(peak, value)
        worst = max(worst, (peak - value) / peak)
    return worst
