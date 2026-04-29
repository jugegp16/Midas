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


def compute_risk_metrics(
    equity_curve: list[tuple[date, float]],
    vol_target: float | None,
    per_strategy_pnl: dict[str, float],
    per_ticker_vol_contribution: dict[str, float] | None = None,
) -> RiskMetrics:
    """Compute current risk metrics from equity curve and pre-aggregated attribution.

    Args:
        equity_curve: list of ``(date, portfolio_value)``, oldest first.
        vol_target: configured target for context (None when disabled).
        per_strategy_pnl: cumulative attributed P&L per strategy name.
        per_ticker_vol_contribution: per-ticker share of portfolio vol (optional).

    Returns:
        ``RiskMetrics`` snapshot at the latest equity-curve point. An empty
        curve yields a zero-everything snapshot.
    """
    if not equity_curve:
        return RiskMetrics(
            realized_vol_60d=0.0,
            vol_target=vol_target,
            drawdown_from_peak=0.0,
            rolling_sharpe_252d=0.0,
            per_strategy_pnl=dict(per_strategy_pnl),
            per_ticker_vol_contribution=per_ticker_vol_contribution or {},
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
    )


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
