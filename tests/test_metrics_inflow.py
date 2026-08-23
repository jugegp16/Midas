"""Tests for inflow-adjusted returns and the metrics built on them."""

from datetime import date, timedelta

import pytest

from midas.metrics import (
    compute_block_returns,
    compute_inflow_adjusted_returns,
    compute_ulcer_index,
)


def _curve(values: list[float]) -> list[tuple[date, float]]:
    return [(date(2024, 1, 1) + timedelta(days=i), v) for i, v in enumerate(values)]


def test_inflow_adjusted_returns_strip_deposits() -> None:
    curve = _curve([100.0, 110.0, 121.0])
    inflows = {date(2024, 1, 2): 10.0}
    returns = compute_inflow_adjusted_returns(curve, inflows)
    assert returns[0] == pytest.approx(0.0)  # the +10 was a deposit, not return
    assert returns[1] == pytest.approx(0.10)


def test_inflow_adjusted_returns_without_inflows_are_plain_returns() -> None:
    returns = compute_inflow_adjusted_returns(_curve([100.0, 90.0, 99.0]), {})
    assert returns == [pytest.approx(-0.10), pytest.approx(0.10)]


def test_ulcer_index_is_rms_of_drawdown_depths() -> None:
    # Curve 100 -> 90 -> 100 -> 100: drawdowns 0%, 10%, 0%, 0%.
    returns = compute_inflow_adjusted_returns(_curve([100.0, 90.0, 100.0, 100.0]), {})
    assert compute_ulcer_index(returns) == pytest.approx((0.01 / 4) ** 0.5)


def test_ulcer_index_of_monotonic_rise_is_zero() -> None:
    assert compute_ulcer_index([0.01, 0.02, 0.005]) == 0.0


def test_block_returns_compound_within_blocks() -> None:
    returns = [0.10, 0.10, -0.10, -0.10]
    blocks = compute_block_returns(returns, n_blocks=2)
    assert blocks == [pytest.approx(1.1 * 1.1 - 1), pytest.approx(0.9 * 0.9 - 1)]


def test_block_returns_spread_remainder_bars() -> None:
    blocks = compute_block_returns([0.01] * 10, n_blocks=4)
    assert len(blocks) == 4
    # 10 bars into 4 blocks: sizes 3,3,2,2 — every bar lands in exactly one block.
    total = 1.0
    for block in blocks:
        total *= 1 + block
    assert total == pytest.approx(1.01**10)


def test_block_returns_with_fewer_bars_than_blocks() -> None:
    assert compute_block_returns([0.05], n_blocks=8) == [pytest.approx(0.05)]
