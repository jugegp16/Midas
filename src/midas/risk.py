"""Risk-discipline primitives: realized vol, covariance, IDM, vol targeting.

Pure functions only — no class state. All functions return new objects
(dicts, DataFrames) and never mutate their inputs. Degenerate inputs are
handled by returning a sentinel (``None`` or an unchanged result) rather
than raising, so the allocator can degrade gracefully at runtime.

See docs/superpowers/specs/2026-04-20-risk-discipline-design.md.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf  # type: ignore[import-untyped]

from midas.data.price_history import PriceHistory

TRADING_DAYS_PER_YEAR = 252


def realized_vol(
    history: PriceHistory,
    window: int,
    annualize: bool = True,
) -> float | None:
    """Realized volatility over the last ``window`` closes.

    Args:
        history: OHLCV bars for a single ticker.
        window: Number of most-recent bars to consider. The vol is computed
            over ``window - 1`` daily simple returns.
        annualize: Multiply by sqrt(252) when True.

    Returns:
        Volatility as a fraction (0.20 == 20% annualized), or None when
        there is insufficient clean history (fewer than ``window`` bars or
        any NaN inside the window).
    """
    if window < 2:
        msg = f"window must be >= 2, got {window}"
        raise ValueError(msg)
    if len(history) < window:
        return None
    closes = history.close[-window:]
    if not np.all(np.isfinite(closes)):
        return None
    returns = closes[1:] / closes[:-1] - 1.0
    if returns.size < 2:
        return None
    daily = float(np.std(returns, ddof=1))
    return daily * math.sqrt(TRADING_DAYS_PER_YEAR) if annualize else daily


def covariance_matrix(
    histories: Mapping[str, PriceHistory],
    vol_window: int,
    corr_window: int,
    vol_floor: float,
) -> pd.DataFrame | None:
    """Composed covariance: vols (short window) x corr (long window, shrunk).

    Correlation is fit with :class:`sklearn.covariance.LedoitWolf` on the
    last ``corr_window`` returns across the universe. Per-ticker vols are
    realized over ``vol_window`` and floored at ``vol_floor`` before the
    outer-product composition.

    Args:
        histories: Mapping of ticker symbol to PriceHistory.
        vol_window: Number of bars for realized vol calculation.
        corr_window: Number of returns for LedoitWolf correlation fit.
        vol_floor: Minimum annualized vol applied after realized vol computation.

    Returns:
        NxN covariance DataFrame indexed by ticker (sorted), or None when
        any ticker has fewer than ``corr_window + 1`` clean bars or the
        LedoitWolf fit fails.

    Raises:
        ValueError: If vol_window or corr_window is less than 2.
    """
    if corr_window < 2 or vol_window < 2:
        msg = f"vol_window and corr_window must be >= 2, got {vol_window}, {corr_window}"
        raise ValueError(msg)

    tickers = sorted(histories.keys())
    if len(tickers) < 1:
        return None

    returns_cols: list[np.ndarray] = []
    for ticker in tickers:
        history = histories[ticker]
        if len(history) < corr_window + 1:
            return None
        closes = history.close[-(corr_window + 1) :]
        if not np.all(np.isfinite(closes)):
            return None
        returns_cols.append(closes[1:] / closes[:-1] - 1.0)

    returns_matrix = np.column_stack(returns_cols)  # shape (corr_window, N)

    try:
        lw = LedoitWolf().fit(returns_matrix)
    except Exception:
        return None

    daily_cov = lw.covariance_
    # Derive correlation from LedoitWolf's daily covariance.
    daily_std = np.sqrt(np.diag(daily_cov))
    # Guard against zero diag from a degenerate constant series.
    if not np.all(daily_std > 0):
        return None
    corr = daily_cov / np.outer(daily_std, daily_std)

    # Per-ticker realized (annualized) vol over the short window. None-fallback
    # = floor so the composition always has a defined scale.
    vols: list[float] = []
    for ticker in tickers:
        v = realized_vol(histories[ticker], window=vol_window, annualize=True)
        vols.append(max(v if v is not None else vol_floor, vol_floor))
    vols_arr = np.asarray(vols)

    cov_composed = corr * np.outer(vols_arr, vols_arr)
    # Numerical safety: symmetrize.
    cov_composed = 0.5 * (cov_composed + cov_composed.T)
    return pd.DataFrame(cov_composed, index=tickers, columns=tickers)
