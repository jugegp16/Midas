"""Unit tests for read-only risk telemetry."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np

from midas.risk_metrics import RiskMetrics, compute_risk_metrics


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
