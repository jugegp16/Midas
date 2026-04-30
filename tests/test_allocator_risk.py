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


# ---------------------------------------------------------------------------
# Phase 2: inverse-vol score offset (T-independence is the PR #63 regression)
# ---------------------------------------------------------------------------


def test_inverse_vol_high_vol_gets_less_weight() -> None:
    """10x vol gap at identical scores → ~10x weight ratio in favour of low-vol."""
    risk = RiskConfig(weighting="inverse_vol", vol_lookback_days=60)
    # max_position_pct must be high enough that the soft cap doesn't bind and
    # mask the inverse-vol effect (LO would otherwise get ~95% of the budget).
    constraints = AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.95)
    alloc = _build_allocator(risk_config=risk, constraints=constraints)
    prices = {
        "LO": _diverging_history(120, daily_vol=0.005, seed=1),
        "HI": _diverging_history(120, daily_vol=0.05, seed=2),
    }
    result = alloc.allocate(["LO", "HI"], prices)
    ratio = result.targets["LO"] / max(result.targets["HI"], 1e-12)
    assert 7.0 < ratio < 13.0


def test_inverse_vol_t_independence() -> None:
    """The offset is added *outside* /T, so the inverse-vol weight ratio at
    identical scores is invariant to softmax_temperature.

    Regression for PR #63 — that bug used (1/vol)^(1/T), which would produce
    wildly different ratios at T=0.2 vs T=2.0 (~32:1 vs ~1.4:1 instead of ~10:1).
    """
    prices = {
        "LO": _diverging_history(120, daily_vol=0.005, seed=1),
        "HI": _diverging_history(120, daily_vol=0.05, seed=2),
    }

    def ratio_at_t(temperature: float) -> float:
        constraints = AllocationConstraints(
            min_cash_pct=0.05,
            softmax_temperature=temperature,
            max_position_pct=0.95,
        )
        alloc = _build_allocator(
            risk_config=RiskConfig(weighting="inverse_vol", vol_lookback_days=60),
            constraints=constraints,
        )
        result = alloc.allocate(["LO", "HI"], prices)
        return result.targets["LO"] / max(result.targets["HI"], 1e-12)

    r_cold = ratio_at_t(0.2)
    r_hot = ratio_at_t(2.0)
    # Identical scores → score term cancels at any T → ratio comes from the
    # offset alone, which is T-independent.
    assert math.isclose(r_cold, r_hot, rel_tol=0.10)


def test_equal_weighting_unaffected_by_vol() -> None:
    """Default weighting='equal' ignores vol entirely."""
    alloc = _build_allocator(risk_config=RiskConfig())
    prices = {
        "LO": _diverging_history(120, daily_vol=0.005, seed=1),
        "HI": _diverging_history(120, daily_vol=0.05, seed=2),
    }
    result = alloc.allocate(["LO", "HI"], prices)
    # Equal scores + equal weighting → equal targets.
    assert math.isclose(result.targets["LO"], result.targets["HI"], rel_tol=1e-9)


def test_inverse_vol_falls_back_when_vol_is_zero() -> None:
    """A constant-price ticker has zero realized vol → falls back to Option A (held)."""
    risk = RiskConfig(weighting="inverse_vol", vol_lookback_days=60)
    constraints = AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.95)
    alloc = _build_allocator(risk_config=risk, constraints=constraints)
    prices = {
        "FLAT": _flat_history(120),
        "HI": _diverging_history(120, daily_vol=0.05, seed=2),
    }
    current = {"FLAT": 0.10, "HI": 0.0}
    result = alloc.allocate(["FLAT", "HI"], prices, current_weights=current)
    # FLAT held at its current weight; HI consumes the active budget.
    assert math.isclose(result.targets["FLAT"], 0.10, abs_tol=1e-9)
    investable = 1.0 - 0.05
    assert math.isclose(sum(result.targets.values()), investable, abs_tol=1e-6)


def test_inverse_vol_insufficient_history_falls_back() -> None:
    """Tickers with fewer than vol_lookback_days+1 bars fall back to held."""
    risk = RiskConfig(weighting="inverse_vol", vol_lookback_days=60)
    constraints = AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.95)
    alloc = _build_allocator(risk_config=risk, constraints=constraints)
    prices = {
        "SHORT": _diverging_history(30, daily_vol=0.05, seed=1),  # < lookback+1
        "FULL": _diverging_history(120, daily_vol=0.05, seed=2),
    }
    current = {"SHORT": 0.10, "FULL": 0.0}
    result = alloc.allocate(["SHORT", "FULL"], prices, current_weights=current)
    assert math.isclose(result.targets["SHORT"], 0.10, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Phase 4: portfolio vol target
# ---------------------------------------------------------------------------


def test_phase_4_vol_target_scales_when_predicted_exceeds() -> None:
    """High-vol portfolio with tight target → all weights scale way down."""
    risk = RiskConfig(weighting="equal", vol_lookback_days=60, vol_target=0.10)
    constraints = AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.95)
    alloc = _build_allocator(risk_config=risk, constraints=constraints)
    prices = {
        "HI1": _diverging_history(120, daily_vol=0.05, seed=1),
        "HI2": _diverging_history(120, daily_vol=0.05, seed=2),
    }
    result = alloc.allocate(["HI1", "HI2"], prices)
    investable = 1.0 - 0.05
    # Predicted vol on a ~80% annualized portfolio with 0.10 target → ~12%
    # of investable. Sum drops well below half investable.
    assert sum(result.targets.values()) < investable * 0.5


def test_phase_4_no_scale_when_below_target() -> None:
    """Low-vol portfolio under target → no scaling."""
    risk = RiskConfig(weighting="equal", vol_lookback_days=60, vol_target=0.50)
    constraints = AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.95)
    alloc = _build_allocator(risk_config=risk, constraints=constraints)
    prices = {
        "LO1": _diverging_history(120, daily_vol=0.005, seed=1),
        "LO2": _diverging_history(120, daily_vol=0.005, seed=2),
    }
    result = alloc.allocate(["LO1", "LO2"], prices)
    investable = 1.0 - 0.05
    assert math.isclose(sum(result.targets.values()), investable, abs_tol=1e-6)


def test_phase_4_skips_on_insufficient_history() -> None:
    """If any active ticker lacks enough history, the entire Phase 4 step is skipped."""
    risk = RiskConfig(weighting="equal", vol_lookback_days=60, vol_target=0.10)
    constraints = AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.95)
    alloc = _build_allocator(risk_config=risk, constraints=constraints)
    prices = {
        "FULL": _diverging_history(120, daily_vol=0.05, seed=1),
        "SHORT": _diverging_history(30, daily_vol=0.05, seed=2),  # < lookback + 1
    }
    result = alloc.allocate(["FULL", "SHORT"], prices)
    # No vol scaling fires; both tickers active; sum = investable.
    investable = 1.0 - 0.05
    assert math.isclose(sum(result.targets.values()), investable, abs_tol=1e-6)


def test_phase_0_phase_4_compose_when_only_phase_0_binds() -> None:
    """With a loose vol_target that Phase 4 doesn't activate, the CPPI overlay
    drives the final gross alone — its 0.7x multiplier reaches the output.

    (When Phase 4 binds it normalizes predicted vol to target, mathematically
    erasing any prior gross-scaling. The two phases compose multiplicatively
    only while Phase 4 is non-binding.)
    """
    risk = RiskConfig(
        weighting="equal",
        vol_lookback_days=60,
        vol_target=2.0,  # 200% — Phase 4 cannot bind on a long-only equity book
        drawdown_penalty=1.5,
        drawdown_floor=0.5,
    )
    constraints = AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.95)
    alloc = _build_allocator(risk_config=risk, constraints=constraints)
    prices = {
        "HI1": _diverging_history(120, daily_vol=0.05, seed=1),
        "HI2": _diverging_history(120, daily_vol=0.05, seed=2),
    }
    result_no_dd = alloc.allocate(["HI1", "HI2"], prices, current_drawdown=0.0)
    result_dd = alloc.allocate(["HI1", "HI2"], prices, current_drawdown=0.20)
    sum_no_dd = sum(result_no_dd.targets.values())
    sum_dd = sum(result_dd.targets.values())
    assert math.isclose(sum_dd, sum_no_dd * 0.70, rel_tol=1e-6)


def test_phase_0_phase_4_both_reduce_only() -> None:
    """Sum with both phases enabled <= sum with neither. Both are reduce-only."""
    base_risk = RiskConfig(weighting="equal")
    full_risk = RiskConfig(
        weighting="equal",
        vol_lookback_days=60,
        vol_target=0.10,
        drawdown_penalty=1.5,
        drawdown_floor=0.5,
    )
    constraints = AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.95)
    base = _build_allocator(risk_config=base_risk, constraints=constraints)
    full = _build_allocator(risk_config=full_risk, constraints=constraints)
    prices = {
        "HI1": _diverging_history(120, daily_vol=0.05, seed=1),
        "HI2": _diverging_history(120, daily_vol=0.05, seed=2),
    }
    base_sum = sum(base.allocate(["HI1", "HI2"], prices, current_drawdown=0.20).targets.values())
    full_sum = sum(full.allocate(["HI1", "HI2"], prices, current_drawdown=0.20).targets.values())
    assert full_sum <= base_sum + 1e-9


def test_phase_4_single_active_ticker_n_eq_1() -> None:
    """Spec line 124: Phase 4 fires for any N >= 1; a 1x1 covariance is well-defined."""
    risk = RiskConfig(weighting="equal", vol_lookback_days=60, vol_target=0.10)
    constraints = AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.95)
    alloc = _build_allocator(risk_config=risk, constraints=constraints, n_tickers=1)
    prices = {"SOLO": _diverging_history(120, daily_vol=0.05, seed=1)}
    result = alloc.allocate(["SOLO"], prices)
    investable = 1.0 - 0.05
    # High-vol single-asset portfolio with tight target → vol target binds.
    assert result.targets["SOLO"] < investable * 0.5


def test_phase_4_held_positions_pass_through_unchanged() -> None:
    """Spec line 122: held tickers' weights are not scaled when Phase 4 binds."""
    risk = RiskConfig(weighting="equal", vol_lookback_days=60, vol_target=0.10)
    constraints = AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.95)
    alloc = Allocator(
        [(_ConstantScore(0.0), 1.0)],  # scores 0 → all tickers HELD via Option A
        constraints,
        n_tickers=2,
        risk_config=risk,
    )
    prices = {
        "H1": _diverging_history(120, daily_vol=0.05, seed=1),
        "H2": _diverging_history(120, daily_vol=0.05, seed=2),
    }
    current = {"H1": 0.30, "H2": 0.20}
    result = alloc.allocate(["H1", "H2"], prices, current_weights=current)
    # Held weights pass through unchanged; Phase 4 sees no active and is a no-op.
    assert math.isclose(result.targets["H1"], 0.30, abs_tol=1e-9)
    assert math.isclose(result.targets["H2"], 0.20, abs_tol=1e-9)


def test_all_active_tickers_hit_inverse_vol_fallback_holds_all() -> None:
    """Spec line 108: every active hits Phase 2 fallback → all held; Phase 4 skipped; no crash."""
    risk = RiskConfig(weighting="inverse_vol", vol_lookback_days=60, vol_target=0.10)
    constraints = AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.95)
    alloc = _build_allocator(risk_config=risk, constraints=constraints)
    # Both tickers have insufficient history → fall back to Option A.
    prices = {
        "S1": _diverging_history(30, daily_vol=0.05, seed=1),
        "S2": _diverging_history(30, daily_vol=0.05, seed=2),
    }
    current = {"S1": 0.20, "S2": 0.10}
    result = alloc.allocate(["S1", "S2"], prices, current_weights=current)
    assert math.isclose(result.targets["S1"], 0.20, abs_tol=1e-9)
    assert math.isclose(result.targets["S2"], 0.10, abs_tol=1e-9)


def test_phase_0_min_cash_pct_compose_multiplicatively() -> None:
    """Spec line 90: investable = (1 - min_cash_pct) * exposure_scale, never the
    other way. With min_cash_pct=0.05 + drawdown_floor=0.5 + 50% DD, investable
    is 0.95 * 0.5 = 0.475, not 0.5 - 0.05 = 0.45.
    """
    risk = RiskConfig(drawdown_penalty=1.5, drawdown_floor=0.5)
    constraints = AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.95)
    alloc = _build_allocator(risk_config=risk, constraints=constraints)
    prices = {"A": _flat_history(60), "B": _flat_history(60)}
    result = alloc.allocate(["A", "B"], prices, current_drawdown=0.50)
    expected_investable = (1.0 - 0.05) * 0.50  # = 0.475
    assert math.isclose(sum(result.targets.values()), expected_investable, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Per-bar risk telemetry on AllocationResult
# ---------------------------------------------------------------------------


def test_telemetry_inert_when_no_risk_config() -> None:
    """Without risk_config, telemetry defaults are all inert (1.0/0.0/False)."""
    alloc = _build_allocator(risk_config=None)
    prices = {"A": _flat_history(60), "B": _flat_history(60)}
    result = alloc.allocate(["A", "B"], prices)
    tel = result.risk_telemetry
    assert tel.cppi_scale == 1.0
    assert tel.vol_target_scale == 1.0
    assert tel.vol_target_skipped is False
    assert tel.predicted_vol == 0.0
    assert math.isclose(tel.gross_exposure, sum(result.targets.values()), abs_tol=1e-9)


def test_telemetry_records_cppi_scale() -> None:
    risk = RiskConfig(drawdown_penalty=1.5, drawdown_floor=0.5)
    alloc = _build_allocator(risk_config=risk)
    prices = {"A": _flat_history(60), "B": _flat_history(60)}
    result = alloc.allocate(["A", "B"], prices, current_drawdown=0.20)
    # 1 - 1.5 * 0.20 = 0.70.
    assert math.isclose(result.risk_telemetry.cppi_scale, 0.70, abs_tol=1e-9)


def test_telemetry_records_vol_target_bind() -> None:
    """When Phase 4 binds, telemetry captures predicted_vol and the scale."""
    risk = RiskConfig(vol_target=0.05, vol_lookback_days=60)
    alloc = _build_allocator(risk_config=risk)
    prices = {
        "A": _diverging_history(80, daily_vol=0.03, seed=1),
        "B": _diverging_history(80, daily_vol=0.03, seed=2),
    }
    result = alloc.allocate(["A", "B"], prices)
    tel = result.risk_telemetry
    # Daily vol 0.03 → annualized ≈ 0.476 — well above the 0.05 target.
    assert tel.predicted_vol > 0.05
    assert tel.vol_target_scale < 1.0
    assert tel.vol_target_skipped is False


def test_telemetry_flags_vol_target_skip() -> None:
    """Constant-price ticker (zero stdev) → Phase 4 skips silently."""
    risk = RiskConfig(vol_target=0.10, vol_lookback_days=60)
    alloc = _build_allocator(risk_config=risk)
    prices = {"A": _flat_history(80), "B": _flat_history(80)}
    result = alloc.allocate(["A", "B"], prices)
    tel = result.risk_telemetry
    assert tel.vol_target_skipped is True
    assert tel.vol_target_scale == 1.0


def test_telemetry_gross_exposure_matches_target_sum() -> None:
    """gross_exposure on telemetry equals sum of final targets — invariant."""
    risk = RiskConfig(drawdown_penalty=1.5, drawdown_floor=0.5)
    alloc = _build_allocator(risk_config=risk)
    prices = {"A": _flat_history(60), "B": _flat_history(60)}
    result = alloc.allocate(["A", "B"], prices, current_drawdown=0.30)
    assert math.isclose(result.risk_telemetry.gross_exposure, sum(result.targets.values()), abs_tol=1e-12)
