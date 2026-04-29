"""Pure risk-discipline functions. No state, no caches.

Inputs are sliced numpy arrays sized at the caller's discretion. Each function
must produce identical output regardless of whether the caller passes a slice
of a larger array or a fresh array — this is the no-lookahead invariant.

All vol quantities returned by this module are annualized via ``sqrt(252)``.
Daily log-return stdev is annualized at the function boundary; daily units
never escape this module.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf  # type: ignore[import-untyped]

TRADING_DAYS_PER_YEAR = 252


def realized_vol(prices: np.ndarray, lookback: int) -> float:
    """Annualized stdev of close-to-close log returns over the last ``lookback`` bars.

    Args:
        prices: 1-D array of closes, oldest first.
        lookback: number of returns (not prices) to consider — the function reads
            the last ``lookback + 1`` prices.

    Returns:
        Annualized stdev. Returns ``0.0`` if fewer than ``lookback + 1`` prices
        are available, or if the log-return series is constant (zero stdev).
        Callers must treat ``0.0`` as "insufficient signal" and fall back
        accordingly.
    """
    if prices.size < lookback + 1:
        return 0.0
    window = prices[-(lookback + 1) :]
    log_returns = np.diff(np.log(window))
    if log_returns.size < 2:
        return 0.0
    return float(np.std(log_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def covariance_matrix(log_returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf-shrunk covariance of daily log returns.

    Args:
        log_returns: shape ``(n_bars, n_tickers)``. Each column is one ticker's
            daily log-return series over a common window.

    Returns:
        ``(n_tickers, n_tickers)`` shrunk covariance matrix in daily units.
        Annualize at the call site if needed.
    """
    estimator = LedoitWolf().fit(log_returns)
    return np.asarray(estimator.covariance_)


def predict_portfolio_vol(weights: np.ndarray, log_returns: np.ndarray) -> float:
    """Annualized predicted portfolio volatility from weights and log-return history.

    Args:
        weights: shape ``(n_tickers,)``. Same column order as ``log_returns``.
        log_returns: shape ``(n_bars, n_tickers)``. Daily log returns over a
            common window.

    Returns:
        Annualized stdev of the portfolio's daily returns under the given
        weights. Returns ``0.0`` when the implied daily variance is non-positive
        (e.g., zero weights, or pathological covariance).
    """
    cov = covariance_matrix(log_returns)
    daily_var = float(weights @ cov @ weights)
    if daily_var <= 0:
        return 0.0
    return float(np.sqrt(daily_var) * np.sqrt(TRADING_DAYS_PER_YEAR))


def apply_drawdown_overlay(current_drawdown: float, penalty: float, floor: float) -> float:
    """CPPI-style exposure scaler: ``max(1 - penalty * dd, floor)``.

    Args:
        current_drawdown: positive fraction (e.g. ``0.20`` for 20% drawdown
            from the running peak).
        penalty: how aggressively to de-risk per unit of drawdown.
        floor: minimum exposure scale; never reduce below this.

    Returns:
        Scalar in ``[floor, 1.0]`` to multiply the gross investable budget by.
    """
    return max(1.0 - penalty * current_drawdown, floor)


def inverse_vol_offset(vol: float, vol_floor: float) -> float:
    """Score offset for inverse-vol weighting: ``-log(max(vol, vol_floor))``.

    Returns ``NaN`` when ``vol == 0``: the caller must treat this as
    "insufficient signal" and fall back (typically Option A: hold the current
    weight). The floor handles low-but-nonzero vols; literal zero indicates
    no valid signal at all.
    """
    if vol == 0.0:
        return float("nan")
    return -float(np.log(max(vol, vol_floor)))
