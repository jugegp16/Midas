"""RSI oversold strategy: buy when RSI drops below threshold."""

from __future__ import annotations

import numpy as np

from midas.data.price_history import PriceHistory
from midas.models import AssetSuitability
from midas.strategies.base import RECURSIVE_WARMUP_MULTIPLIER, EntrySignal

RSI_MIDPOINT = 50.0


def wilder_rsi(prices: np.ndarray, window: int) -> np.ndarray:
    """Compute the Wilder RSI series for *prices*.

    Standard Wilder smoothing: average gain/loss seeded with the plain
    mean of the first *window* deltas, then updated recursively as
    ``avg_t = (avg_{t-1} * (window - 1) + value_t) / window``. This is
    the formulation every charting package and data provider uses — a
    plain rolling mean systematically disagrees with all of them
    (issue #45).

    Args:
        prices: Close prices, oldest first.
        window: RSI period (14 in Wilder's original).

    Returns:
        Array aligned to *prices*: NaN before index ``window`` (not
        enough deltas yet), RSI in [0, 100] from there on. Zero average
        loss yields the conventional RSI of 100.
    """
    num_bars = len(prices)
    rsi = np.full(num_bars, np.nan)
    if num_bars < window + 1:
        return rsi
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(gains[:window].mean())
    avg_loss = float(losses[:window].mean())
    rsi[window] = _rsi_from_averages(avg_gain, avg_loss)
    # deltas[t] is the move INTO price bar t+1, so the recursive update
    # over delta index t produces the RSI at price index t+1.
    for t in range(window, num_bars - 1):
        avg_gain = (avg_gain * (window - 1) + gains[t]) / window
        avg_loss = (avg_loss * (window - 1) + losses[t]) / window
        rsi[t + 1] = _rsi_from_averages(avg_gain, avg_loss)
    return rsi


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


class RSIOversold(EntrySignal):
    """Entry signal: bullish as RSI drops below the 50 midpoint toward oversold."""

    def __init__(self, window: int = 14, oversold_threshold: float = 30.0) -> None:
        self._window = window
        self._oversold_threshold = oversold_threshold

    @property
    def warmup_period(self) -> int:
        # RSI is recursive (Wilder smoothing) — strict min of `window`
        # bars gives a mathematically valid but numerically unstable value.
        return self._window * RECURSIVE_WARMUP_MULTIPLIER

    def precompute(self, price_history: PriceHistory) -> np.ndarray | None:
        rsi = wilder_rsi(price_history.close, self._window)
        scale = RSI_MIDPOINT - self._oversold_threshold
        if scale == 0:
            return np.where(np.isnan(rsi), np.nan, 0.0)
        # RSI >= midpoint maps negative and clips to 0; NaN propagates.
        return np.clip((RSI_MIDPOINT - rsi) / scale, 0.0, 1.0)

    def score(
        self,
        price_history: PriceHistory,
        **kwargs: object,
    ) -> float | None:
        # Wilder RSI is recursive over the full history, so the live path
        # delegates to precompute and reads the final bar — parity with
        # backtest holds by construction instead of by a parallel
        # implementation. O(bars) per tick over the warmup-sized window
        # live feeds it, which is trivial.
        scores = self.precompute(price_history)
        if scores is None or len(scores) == 0 or np.isnan(scores[-1]):
            return None
        return float(scores[-1])

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return f"Bullish when RSI is below 50 ({self._window}-period, oversold at {self._oversold_threshold:.0f})"
