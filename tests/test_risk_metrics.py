"""Unit tests for read-only risk telemetry."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np

from midas.risk_metrics import RiskHistory, RiskMetrics, compute_risk_metrics


def _equity_curve(values: list[float]) -> list[tuple[date, float]]:
    start = date(2020, 1, 1)
    return [(start + timedelta(days=i), v) for i, v in enumerate(values)]


class TestComputeRiskMetrics:
    def test_drawdown_from_peak(self) -> None:
        # Peak 110, current 99 → 10% drawdown.
        curve = _equity_curve([100, 105, 110, 99])
        metrics = compute_risk_metrics(curve, vol_target=None, per_strategy_pnl={})
        assert math.isclose(metrics.drawdown_from_peak, 0.10, abs_tol=1e-9)

    def test_no_drawdown_at_new_peak(self) -> None:
        curve = _equity_curve([100, 110, 120])
        metrics = compute_risk_metrics(curve, vol_target=None, per_strategy_pnl={})
        assert metrics.drawdown_from_peak == 0.0

    def test_realized_vol_60d(self) -> None:
        rng = np.random.default_rng(0)
        # 80 days of synthetic returns with daily stdev 0.01.
        returns = rng.normal(0.0, 0.01, 80)
        values = [100.0]
        for r in returns:
            values.append(values[-1] * (1 + r))
        curve = _equity_curve(values)
        metrics = compute_risk_metrics(curve, vol_target=None, per_strategy_pnl={})
        # Annualized: 0.01 * sqrt(252) ≈ 0.1587, with ±20% sample noise.
        assert math.isclose(metrics.realized_vol_60d, 0.01 * math.sqrt(252), rel_tol=0.20)

    def test_per_strategy_pnl_passthrough(self) -> None:
        curve = _equity_curve([100, 110])
        metrics = compute_risk_metrics(
            curve,
            vol_target=0.20,
            per_strategy_pnl={"BollingerBand": 5.0, "RSIOversold": 5.0},
        )
        assert metrics.per_strategy_pnl == {"BollingerBand": 5.0, "RSIOversold": 5.0}
        assert metrics.vol_target == 0.20

    def test_empty_equity_curve(self) -> None:
        metrics = compute_risk_metrics([], vol_target=None, per_strategy_pnl={})
        assert metrics == RiskMetrics(
            realized_vol_60d=0.0,
            vol_target=None,
            drawdown_from_peak=0.0,
            rolling_sharpe_252d=0.0,
            per_strategy_pnl={},
            per_ticker_vol_contribution={},
        )

    def test_inert_aggregates_when_no_history(self) -> None:
        metrics = compute_risk_metrics(_equity_curve([100, 101]), vol_target=None, per_strategy_pnl={})
        assert metrics.cppi_active_pct == 0.0
        assert metrics.cppi_avg_scale == 1.0
        assert metrics.cppi_min_scale == 1.0
        assert metrics.vol_target_bind_pct == 0.0
        assert metrics.vol_target_avg_scale == 1.0
        assert metrics.vol_target_skip_count == 0
        assert metrics.avg_gross_exposure == 0.0
        assert metrics.min_gross_exposure == 0.0


class TestPhaseAggregates:
    def _history(
        self,
        cppi_scale: list[float],
        vol_target_scale: list[float],
        gross_exposure: list[float],
        vol_target_predicted_vol: list[float] | None = None,
    ) -> RiskHistory:
        n = len(cppi_scale)
        start = date(2020, 1, 1)
        return RiskHistory(
            dates=[start + timedelta(days=i) for i in range(n)],
            gross_exposure=gross_exposure,
            cppi_scale=cppi_scale,
            vol_target_scale=vol_target_scale,
            vol_target_predicted_vol=vol_target_predicted_vol if vol_target_predicted_vol is not None else [0.0] * n,
            drawdown=[0.0] * n,
        )

    def test_cppi_active_pct(self) -> None:
        # 4 of 10 bars under 1.0 → 40% active.
        history = self._history(
            cppi_scale=[1.0, 0.8, 1.0, 0.7, 1.0, 0.6, 1.0, 1.0, 1.0, 0.5],
            vol_target_scale=[1.0] * 10,
            gross_exposure=[0.95] * 10,
        )
        metrics = compute_risk_metrics(
            _equity_curve([100.0]),
            vol_target=None,
            per_strategy_pnl={},
            risk_history=history,
        )
        assert math.isclose(metrics.cppi_active_pct, 0.4, abs_tol=1e-9)
        assert math.isclose(metrics.cppi_min_scale, 0.5, abs_tol=1e-9)

    def test_vol_target_bind_pct_and_skip_count(self) -> None:
        history = self._history(
            cppi_scale=[1.0] * 10,
            vol_target_scale=[1.0, 0.8, 0.7, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0],
            gross_exposure=[0.95] * 10,
        )
        metrics = compute_risk_metrics(
            _equity_curve([100.0]),
            vol_target=0.10,
            per_strategy_pnl={},
            risk_history=history,
            vol_target_skip_count=3,
        )
        # 3 of 10 bars bound (scale<1.0).
        assert math.isclose(metrics.vol_target_bind_pct, 0.3, abs_tol=1e-9)
        assert metrics.vol_target_skip_count == 3

    def test_gross_exposure_aggregates(self) -> None:
        history = self._history(
            cppi_scale=[1.0] * 5,
            vol_target_scale=[1.0] * 5,
            gross_exposure=[0.95, 0.80, 0.70, 0.60, 0.50],
        )
        metrics = compute_risk_metrics(
            _equity_curve([100.0]),
            vol_target=None,
            per_strategy_pnl={},
            risk_history=history,
        )
        assert math.isclose(metrics.avg_gross_exposure, 0.71, abs_tol=1e-9)
        assert math.isclose(metrics.min_gross_exposure, 0.50, abs_tol=1e-9)
