"""Ensemble deployment helpers.

An ensemble is K entry-parameter members whose blended scores are averaged
before one policy-level sizing/risk/exit pass. This module holds the pure
helpers; the allocator owns the blending and the fitter owns member
selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

BEHAVIORAL_DECIMALS = 4


def behavioral_key(
    target_series: Sequence[Mapping[str, float]],
) -> tuple[tuple[tuple[str, float], ...], ...]:
    """Hashable fingerprint of a per-bar target-weight series.

    Two parameter tuples that differ only in a never-binding parameter
    produce identical targets and must count as one member — dedupe is
    behavioral, not textual. Weights are rounded so float jitter below
    the rounding threshold cannot defeat the collapse.
    """
    return tuple(
        tuple(sorted((ticker, round(weight, BEHAVIORAL_DECIMALS)) for ticker, weight in bar.items()))
        for bar in target_series
    )
