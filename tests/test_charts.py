"""Smoke tests for the terminal charts module.

These tests exercise the render path with realistic shapes; they assert the
function doesn't crash and produces some non-empty output. The goal is
regression protection against shape mismatches between ``RiskHistory`` and
``equity_curve``, not pixel-level chart correctness.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from midas.charts import render_charts
from midas.results import BacktestResult
from midas.risk_metrics import RiskHistory, RiskMetrics


def _empty_result() -> BacktestResult:
    return BacktestResult(
        trades=[],
        final_value=0,
        starting_value=0,
        buy_and_hold_value=0,
        train_trades=[],
        test_trades=[],
        train_return=0,
        test_return=0,
        train_bh_return=0,
        test_bh_return=0,
        split_date=None,
        twr=0,
        equity_curve=[],
        total_days=0,
        train_days=0,
        test_days=0,
        cagr=0,
        max_drawdown=0,
        sharpe_ratio=0,
        sortino_ratio=0,
        win_rate=0,
        profit_factor=0,
        avg_win=0,
        avg_loss=0,
        efficiency_ratio=0,
        strategy_stats=[],
        unrealized_pnl=0,
        unrealized_pnl_by_ticker={},
        basis_per_sell=[],
    )


def _populated_result(*, vol_target: float | None = None, with_history: bool = True) -> BacktestResult:
    n = 60
    start = date(2024, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n)]
    equity = [100.0 + i for i in range(n)]
    history: RiskHistory | None = None
    if with_history:
        history = RiskHistory(
            dates=list(dates),
            gross_exposure=[0.95] * n,
            cppi_scale=[1.0 if i < 30 else 0.85 for i in range(n)],
            vol_target_scale=[1.0 if vol_target is None else 0.9] * n,
            predicted_vol=[0.0 if vol_target is None else 0.12] * n,
            drawdown=[0.0] * n,
        )
    metrics = RiskMetrics(
        realized_vol_60d=0.15,
        vol_target=vol_target,
        drawdown_from_peak=0.0,
        rolling_sharpe_252d=1.2,
    )
    result = _empty_result()
    result.equity_curve = list(zip(dates, equity, strict=True))
    result.risk_metrics = metrics
    result.risk_history = history
    return result


def test_render_charts_no_crash_on_empty_curve() -> None:
    render_charts(_empty_result())


def test_render_charts_no_crash_without_history(capsys: pytest.CaptureFixture[str]) -> None:
    result = _populated_result(with_history=False)
    render_charts(result)
    out = capsys.readouterr().out
    assert "Equity Curve" in out


def test_render_charts_with_cppi_history(capsys: pytest.CaptureFixture[str]) -> None:
    result = _populated_result()
    render_charts(result)
    out = capsys.readouterr().out
    assert "Equity Curve" in out
    assert "Drawdown" in out
    assert "Gross Exposure" in out


def test_render_charts_with_vol_target_panel(capsys: pytest.CaptureFixture[str]) -> None:
    result = _populated_result(vol_target=0.10)
    render_charts(result)
    out = capsys.readouterr().out
    assert "Predicted vs Target" in out
