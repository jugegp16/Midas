"""Read-only risk telemetry. No feedback into construction.

Computed each bar in backtest, each tick in live. Surfaced through the
existing output layer; never modifies allocator state or trade decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from midas.risk import TRADING_DAYS_PER_YEAR

VOL_LOOKBACK_BARS = 60
SHARPE_LOOKBACK_BARS = 252


@dataclass(frozen=True)
class RiskMetrics:
    """Snapshot of portfolio risk state. Strictly observational."""

    realized_vol_60d: float
    vol_target: float | None
    drawdown_from_peak: float
    rolling_sharpe_252d: float
    per_strategy_pnl: dict[str, float] = field(default_factory=dict)
    per_ticker_vol_contribution: dict[str, float] = field(default_factory=dict)
    # Risk-engine activity aggregates over the run. All default to inert
    # values when the corresponding mechanic is disabled, so downstream
    # consumers can render unconditionally.
    cppi_active_pct: float = 0.0
    cppi_avg_scale: float = 1.0
    cppi_min_scale: float = 1.0
    vol_target_bind_pct: float = 0.0
    vol_target_avg_scale: float = 1.0
    vol_target_skip_count: int = 0
    avg_gross_exposure: float = 0.0
    min_gross_exposure: float = 0.0


@dataclass(frozen=True)
class RiskHistory:
    """Per-bar risk telemetry across the backtest, parallel arrays.

    Empty when the backtest produced no equity-curve points. Otherwise every
    list has length ``len(dates)``. Populated unconditionally — when a phase
    is disabled its array is a flat line of inert values (``cppi_scale=1.0``,
    ``vol_target_scale=1.0``, ``predicted_vol=0.0``).
    """

    dates: list[date] = field(default_factory=list)
    gross_exposure: list[float] = field(default_factory=list)
    cppi_scale: list[float] = field(default_factory=list)
    vol_target_scale: list[float] = field(default_factory=list)
    predicted_vol: list[float] = field(default_factory=list)
    drawdown: list[float] = field(default_factory=list)


def compute_risk_metrics(
    equity_curve: list[tuple[date, float]],
    vol_target: float | None,
    per_strategy_pnl: dict[str, float],
    per_ticker_vol_contribution: dict[str, float] | None = None,
    risk_history: RiskHistory | None = None,
    vol_target_skip_count: int = 0,
) -> RiskMetrics:
    """Compute current risk metrics from equity curve and pre-aggregated attribution.

    Args:
        equity_curve: list of ``(date, portfolio_value)``, oldest first.
        vol_target: configured target for context (None when disabled).
        per_strategy_pnl: cumulative attributed P&L per strategy name.
        per_ticker_vol_contribution: per-ticker share of portfolio vol (optional).
        risk_history: per-bar telemetry; when present, drives the phase-activity
            aggregates (CPPI active fraction, vol-target bind fraction, gross
            exposure summary stats).
        vol_target_skip_count: number of bars on which Phase 4b was configured
            but skipped silently (insufficient history / non-positive close /
            zero stdev). Surfaced raw on ``RiskMetrics`` so users can tell
            whether vol-target was *configured but inert*.

    Returns:
        ``RiskMetrics`` snapshot at the latest equity-curve point. An empty
        curve yields a zero-everything snapshot.
    """
    aggregates = _aggregate_history(risk_history)

    if not equity_curve:
        return RiskMetrics(
            realized_vol_60d=0.0,
            vol_target=vol_target,
            drawdown_from_peak=0.0,
            rolling_sharpe_252d=0.0,
            per_strategy_pnl=dict(per_strategy_pnl),
            per_ticker_vol_contribution=per_ticker_vol_contribution or {},
            vol_target_skip_count=vol_target_skip_count,
            **aggregates,
        )

    values = np.array([v for _, v in equity_curve], dtype=float)

    peak = float(np.maximum.accumulate(values)[-1])
    current = float(values[-1])
    drawdown = (peak - current) / peak if peak > 0 else 0.0

    realized_vol = _annualized_vol(values, VOL_LOOKBACK_BARS)
    sharpe = _rolling_sharpe(values, SHARPE_LOOKBACK_BARS)

    return RiskMetrics(
        realized_vol_60d=realized_vol,
        vol_target=vol_target,
        drawdown_from_peak=drawdown,
        rolling_sharpe_252d=sharpe,
        per_strategy_pnl=dict(per_strategy_pnl),
        per_ticker_vol_contribution=per_ticker_vol_contribution or {},
        vol_target_skip_count=vol_target_skip_count,
        **aggregates,
    )


def _aggregate_history(history: RiskHistory | None) -> dict[str, float]:
    """Reduce per-bar telemetry into the scalar aggregates on ``RiskMetrics``.

    Returns inert defaults (``cppi_avg_scale=1.0``, etc.) when ``history`` is
    ``None`` or empty so callers can splat the result regardless of whether
    the engine populated history.
    """
    if history is None or not history.dates:
        return {
            "cppi_active_pct": 0.0,
            "cppi_avg_scale": 1.0,
            "cppi_min_scale": 1.0,
            "vol_target_bind_pct": 0.0,
            "vol_target_avg_scale": 1.0,
            "avg_gross_exposure": 0.0,
            "min_gross_exposure": 0.0,
        }
    cppi = np.asarray(history.cppi_scale, dtype=float)
    vol_scale = np.asarray(history.vol_target_scale, dtype=float)
    gross = np.asarray(history.gross_exposure, dtype=float)
    n = float(len(history.dates))
    return {
        "cppi_active_pct": float(np.sum(cppi < 1.0) / n),
        "cppi_avg_scale": float(np.mean(cppi)),
        "cppi_min_scale": float(np.min(cppi)),
        "vol_target_bind_pct": float(np.sum(vol_scale < 1.0) / n),
        "vol_target_avg_scale": float(np.mean(vol_scale)),
        "avg_gross_exposure": float(np.mean(gross)),
        "min_gross_exposure": float(np.min(gross)),
    }


def _log_returns_window(values: np.ndarray, lookback: int) -> np.ndarray:
    if values.size < 2:
        return np.empty(0)
    window = values[-(lookback + 1) :] if values.size > lookback else values
    positive = window[window > 0]
    if positive.size < 2:
        return np.empty(0)
    return np.diff(np.log(positive))


def _annualized_vol(values: np.ndarray, lookback: int) -> float:
    log_returns = _log_returns_window(values, lookback)
    if log_returns.size < 2:
        return 0.0
    return float(np.std(log_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _rolling_sharpe(values: np.ndarray, lookback: int) -> float:
    log_returns = _log_returns_window(values, lookback)
    if log_returns.size < 2:
        return 0.0
    mean = float(np.mean(log_returns))
    stdev = float(np.std(log_returns, ddof=1))
    if stdev == 0.0:
        return 0.0
    return mean / stdev * float(np.sqrt(TRADING_DAYS_PER_YEAR))
