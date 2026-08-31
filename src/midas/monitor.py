"""Live monitor: rank the trailing fold-window against validation baselines.

Notify-and-act: these are report lines, never engine actions. The window
compares at the same day offset as the validation folds (no annualization
of partial windows), the drawdown anchor is the window peak (matching how
fold drawdowns were measured), and a baselines artifact whose input hash
does not match the running configuration is refused, not reinterpreted.
"""

from __future__ import annotations

from collections.abc import Sequence

from midas.baselines import Baselines


def offset_fold_rank(fold_paths: Sequence[Sequence[float]], offset: int, value: float) -> float | None:
    """Midrank percentile of *value* among the folds' cumulative returns at *offset*.

    This is THE ranking — the monitor prints it and the adoption trigger
    fires on it, so the operator never reads one number while the policy
    acts on another. Fold values are ranked directly (never a mixture
    bootstrap of the concatenated path, which treats a fold-persistent
    regime as an improbable streak): a window riding a validated fold's
    exact rate ranks at that fold's mass, not below it. Ties count half.

    Returns:
        None when no fold is long enough to have a value at *offset*.
    """
    at_offset = [path[offset] for path in fold_paths if len(path) > offset]
    if not at_offset:
        return None
    below = sum(1 for v in at_offset if v < value)
    equal = sum(1 for v in at_offset if v == value)
    return (below + equal / 2.0) * 100.0 / len(at_offset)


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
    fold_paths = [_cumulative_path(fold.daily_returns) for fold in baselines.folds]
    rank = offset_fold_rank(fold_paths, offset, cumulative)
    if rank is not None:
        position = f"p{rank:.0f} of validation folds at this offset"
    else:
        position = "validation distribution unavailable"
    day = len(window_returns)
    grade = "" if baselines.includes_holdout else " (pre-holdout baselines)"
    day_label = f"day {day}/{baselines.cadence_days}"
    if day > baselines.cadence_days:
        # The window outlived the cadence (a re-fit is overdue or was
        # declined) — say so instead of printing day 90/63.
        day_label = f"day {day} (past cadence {baselines.cadence_days})"
    lines = [f"fold-window {day_label}: {cumulative:+.2%} TWR (pre-tax) — {position}{grade}"]

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


def _cumulative_path(daily_returns: Sequence[float]) -> list[float]:
    values: list[float] = []
    value = 1.0
    for ret in daily_returns:
        value *= 1.0 + ret
        values.append(value - 1.0)
    return values
