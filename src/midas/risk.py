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


def apply_instrument_diversification_multiplier(
    weights: dict[str, float],
    corr: pd.DataFrame,
    cap: float,
) -> dict[str, float]:
    """Scale all weights by ``min(1/sqrt(w·corr·w), cap)``.

    Args:
        weights: Ticker -> weight before IDM.
        corr: Symmetric correlation matrix. Tickers in ``weights`` but not
            in ``corr`` are left unscaled (treated as uncorrelated singletons).
        cap: Upper bound on the multiplier. Must be >= 1.0 (checked by
            :class:`RiskConfig`).

    Returns:
        New dict with scaled weights. Returns inputs unchanged on degenerate
        inputs (empty weights, single ticker, zero weights, or numerical
        w·corr·w <= 0).
    """
    if not weights:
        return {}
    if sum(weights.values()) == 0:
        return dict(weights)

    # Intersect weights with corr index — only covered tickers participate.
    covered = [t for t in weights if t in corr.index]
    if len(covered) <= 1:
        return dict(weights)

    w_raw = np.asarray([weights[t] for t in covered])
    w_sum = float(w_raw.sum())
    if w_sum <= 0:
        return dict(weights)
    w = w_raw / w_sum
    sub_corr = corr.loc[covered, covered].values
    quad = float(w @ sub_corr @ w)
    if quad <= 0:
        return dict(weights)

    raw_idm = 1.0 / math.sqrt(quad)
    multiplier = min(raw_idm, cap)
    if multiplier <= 1.0:
        # min(idm, cap) == 1.0 exactly happens only when idm<=1 (perfect
        # correlation) or cap==1.0 (IDM stage disabled). Return unchanged.
        return dict(weights)

    return {t: weight * multiplier if t in set(covered) else weight for t, weight in weights.items()}


def predict_portfolio_vol(
    weights: dict[str, float],
    cov: pd.DataFrame,
) -> float:
    """Ex-ante portfolio vol: ``sqrt(w · cov · w) * sqrt(252)``.

    Tickers in ``weights`` but not in ``cov`` are ignored. Returns 0.0 on
    empty weights, all-zero weights, or when the quadratic form is
    non-positive (degenerate covariance).

    Args:
        weights: Ticker -> weight mapping.
        cov: Annualized covariance matrix indexed by ticker symbol.

    Returns:
        Annualized portfolio volatility as a fraction (0.20 == 20%),
        or 0.0 on degenerate inputs.
    """
    covered = [t for t in weights if t in cov.index]
    if not covered:
        return 0.0
    w = np.asarray([weights[t] for t in covered])
    sub_cov = cov.loc[covered, covered].values
    quad = float(w @ sub_cov @ w)
    if quad <= 0:
        return 0.0
    return math.sqrt(quad)


def apply_vol_targeting(
    weights: dict[str, float],
    cov: pd.DataFrame,
    target_annualized_vol: float,
) -> dict[str, float]:
    """Scale weights down so predicted vol does not exceed the target.

    Never scales up — if predicted vol is below target, weights are
    returned unchanged. Degenerate cov matrices (w·cov·w <= 0) also
    pass through unchanged.

    Args:
        weights: Ticker -> weight mapping before vol targeting.
        cov: Annualized covariance matrix indexed by ticker symbol.
        target_annualized_vol: Maximum allowable annualized portfolio vol
            as a fraction (e.g. 0.20 for 20%).

    Returns:
        New dict with weights scaled by ``target / predicted`` when
        predicted vol exceeds the target; otherwise the original weights
        (as a new dict).
    """
    if not weights:
        return {}
    if sum(weights.values()) == 0:
        return dict(weights)

    predicted = predict_portfolio_vol(weights, cov)
    if predicted <= 0 or predicted <= target_annualized_vol:
        return dict(weights)

    scale = target_annualized_vol / predicted
    return {t: weight * scale for t, weight in weights.items()}
