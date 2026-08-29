"""Live monitor: rank the trailing fold-window against validation baselines.

Notify-and-act: these are report lines, never engine actions. The window
compares at the same day offset as the validation folds (no annualization
of partial windows), the drawdown anchor is the window peak (matching how
fold drawdowns were measured), and a baselines artifact whose input hash
does not match the running configuration is refused, not reinterpreted.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from midas.baselines import Baselines

OFFSET_RESAMPLES = 500
MEAN_BLOCK = 21


def offset_cumulative_distribution(
    fold_returns_concat: Sequence[float],
    offset: int,
    *,
    n_resamples: int = OFFSET_RESAMPLES,
    mean_block: int = MEAN_BLOCK,
    seed: int = 0,
) -> list[float]:
    """Bootstrap cumulative returns at a given day offset.

    Stationary-bootstrap resamples of length ``offset + 1`` drawn from the
    concatenated validation path — a smoother reference than ranking
    against N autocorrelated folds at 1/N granularity.
    """
    n = len(fold_returns_concat)
    if n < 2 or offset < 0:
        return []
    rng = random.Random(seed)
    restart_p = 1.0 / mean_block
    out: list[float] = []
    for _ in range(n_resamples):
        idx = rng.randrange(n)
        growth = 1.0
        for _step in range(offset + 1):
            growth *= 1.0 + fold_returns_concat[idx]
            idx = rng.randrange(n) if rng.random() < restart_p else (idx + 1) % n
        out.append(growth - 1.0)
    return out


def monitor_lines(
    baselines: Baselines,
    window_returns: Sequence[float],
    *,
    current_input_hash: str,
) -> tuple[list[str], bool]:
    """Render the report's monitor lines. Returns ``(lines, drawdown_breached)``."""
    if current_input_hash != baselines.input_hash:
        return (["monitor: baselines stale (input mismatch) — comparison refused"], False)
    if not window_returns:
        return ([], False)

    cumulative = 1.0
    for ret in window_returns:
        cumulative *= 1.0 + ret
    cumulative -= 1.0
    offset = len(window_returns) - 1
    concat = [r for fold in baselines.folds for r in fold.daily_returns]
    dist = sorted(offset_cumulative_distribution(concat, offset))
    if dist:
        rank = sum(1 for v in dist if v < cumulative) * 100 // len(dist)
        position = f"p{rank} of validation distribution"
    else:
        position = "validation distribution unavailable"
    day = len(window_returns)
    lines = [f"fold-window day {day}/{baselines.cadence_days}: {cumulative:+.2%} TWR (pre-tax) — {position}"]

    window_dd = _window_drawdown(window_returns)
    worst = max((fold.drawdown for fold in baselines.folds), default=0.0)
    breached = window_dd > worst
    verdict = "BREACH" if breached else "normal"
    lines.append(f"drawdown (window peak): {window_dd:.2%} vs validated worst-fold {worst:.2%} — {verdict}")
    return (lines, breached)


def _window_drawdown(daily_returns: Sequence[float]) -> float:
    """Max drawdown from the window's own peak (mirrors fold measurement)."""
    value = 1.0
    peak = 1.0
    worst = 0.0
    for ret in daily_returns:
        value *= 1.0 + ret
        peak = max(peak, value)
        worst = max(worst, (peak - value) / peak)
    return worst
