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
