"""Risk-discipline primitives: realized vol, covariance, IDM, vol targeting.

Pure functions only — no class state. All functions return new objects
(dicts, DataFrames) and never mutate their inputs. Degenerate inputs are
handled by returning a sentinel (``None`` or an unchanged result) rather
than raising, so the allocator can degrade gracefully at runtime.

See docs/superpowers/specs/2026-04-20-risk-discipline-design.md.
"""

from __future__ import annotations

import math

import numpy as np

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
