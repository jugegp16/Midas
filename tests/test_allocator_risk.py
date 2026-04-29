"""Allocator integration tests for the risk engine (Phases 0, 2, 4)."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np

from midas.allocator import Allocator
from midas.data.price_history import PriceHistory
from midas.models import AllocationConstraints, AssetSuitability, RiskConfig
from midas.strategies.base import EntrySignal


def _flat_history(n_bars: int, price: float = 100.0) -> PriceHistory:
    arr = np.full(n_bars, price)
    dates = np.asarray(
        [date(2024, 1, 1) + timedelta(days=i) for i in range(n_bars)],
        dtype=object,
    )
    return PriceHistory(dates=dates, open=arr, high=arr, low=arr, close=arr, volume=arr)


def _diverging_history(n_bars: int, daily_vol: float, seed: int) -> PriceHistory:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0, daily_vol, n_bars)
    closes = 100.0 * np.exp(np.cumsum(log_returns))
    dates = np.asarray(
        [date(2024, 1, 1) + timedelta(days=i) for i in range(n_bars)],
        dtype=object,
    )
    return PriceHistory(dates=dates, open=closes, high=closes, low=closes, close=closes, volume=closes)


class _ConstantScore(EntrySignal):
    """Test stub: always scores ``_score`` for any ticker with any history."""

    def __init__(self, fixed_score: float = 1.0) -> None:
        self._fixed_score = fixed_score

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return "test stub: constant score"

    def score(self, price_history: PriceHistory, **_: object) -> float | None:
        return self._fixed_score


def _build_allocator(
    risk_config: RiskConfig | None = None,
    constraints: AllocationConstraints | None = None,
    n_tickers: int = 2,
) -> Allocator:
    # max_position_pct=0.5 keeps the 2-ticker softmax from binding the cap and
    # masking the Phase 0 / Phase 4 effects we're isolating in these tests.
    cons = constraints if constraints is not None else AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.5)
    return Allocator(
        [(_ConstantScore(1.0), 1.0)],
        cons,
        n_tickers=n_tickers,
        risk_config=risk_config,
    )


# ---------------------------------------------------------------------------
# Phase 0: CPPI drawdown overlay
# ---------------------------------------------------------------------------


def test_phase_0_no_drawdown_matches_baseline() -> None:
    """current_drawdown=0 → CPPI is a no-op; sum(targets) == investable."""
    risk = RiskConfig(drawdown_penalty=1.5, drawdown_floor=0.5)
    alloc = _build_allocator(risk_config=risk)
    prices = {"A": _flat_history(60), "B": _flat_history(60)}
    result = alloc.allocate(["A", "B"], prices, current_drawdown=0.0)
    investable = 1.0 - 0.05
    assert math.isclose(sum(result.targets.values()), investable, abs_tol=1e-9)


def test_phase_0_drawdown_shrinks_investable() -> None:
    """20% drawdown * penalty 1.5 → exposure 0.70; sum(targets) = 0.95 * 0.70."""
    risk = RiskConfig(drawdown_penalty=1.5, drawdown_floor=0.5)
    alloc = _build_allocator(risk_config=risk)
    prices = {"A": _flat_history(60), "B": _flat_history(60)}
    result = alloc.allocate(["A", "B"], prices, current_drawdown=0.20)
    investable = (1.0 - 0.05) * 0.70
    assert math.isclose(sum(result.targets.values()), investable, abs_tol=1e-6)


def test_phase_0_floor_binds_at_deep_drawdown() -> None:
    """50% drawdown * penalty 1.5 = 0.25 raw, but floor=0.5 binds."""
    risk = RiskConfig(drawdown_penalty=1.5, drawdown_floor=0.5)
    alloc = _build_allocator(risk_config=risk)
    prices = {"A": _flat_history(60), "B": _flat_history(60)}
    result = alloc.allocate(["A", "B"], prices, current_drawdown=0.50)
    investable = (1.0 - 0.05) * 0.50
    assert math.isclose(sum(result.targets.values()), investable, abs_tol=1e-6)


def test_phase_0_ignored_when_overlay_disabled() -> None:
    """No drawdown_penalty configured → current_drawdown is ignored entirely."""
    alloc = _build_allocator(risk_config=RiskConfig())
    prices = {"A": _flat_history(60), "B": _flat_history(60)}
    result = alloc.allocate(["A", "B"], prices, current_drawdown=0.50)
    investable = 1.0 - 0.05
    assert math.isclose(sum(result.targets.values()), investable, abs_tol=1e-9)
