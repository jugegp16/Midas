"""MACD crossover entry: bullish when MACD line is above signal line.

The bearish "MACD below signal" half lives in ``MACDExit`` — entries and
exits are separate types and never blended together.
"""

from __future__ import annotations

import numpy as np

from midas.models import AssetSuitability
from midas.strategies.base import RECURSIVE_WARMUP_MULTIPLIER, EntrySignal


def ema(values: np.ndarray, period: int) -> np.ndarray:
    """Compute exponential moving average."""
    alpha = 2.0 / (period + 1)
    result = np.empty_like(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


class MACDCrossover(EntrySignal):
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> None:
        self._fast_period = fast_period
        self._slow_period = slow_period
        self._signal_period = signal_period

    @property
    def warmup_period(self) -> int:
        # MACD chains three EMAs (fast, slow, signal). Strict min is
        # slow + signal; give EMA room to converge with the recursive
        # multiplier applied to the slow leg.
        return self._slow_period * RECURSIVE_WARMUP_MULTIPLIER + self._signal_period

    def precompute(self, prices: np.ndarray) -> np.ndarray | None:
        n = len(prices)
        min_len = self._slow_period + self._signal_period
        scores = np.full(n, np.nan)
        if n < min_len:
            return scores
        fast_ema = ema(prices, self._fast_period)
        slow_ema = ema(prices, self._slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = ema(macd_line, self._signal_period)
        diff = macd_line - signal_line
        raw = np.where(prices != 0, diff / prices * 100, 0.0)
        scores[min_len - 1 :] = np.clip(raw[min_len - 1 :], 0.0, 1.0)
        return scores

    def score(
        self,
        price_history: np.ndarray,
        **kwargs: object,
    ) -> float | None:
        min_len = self._slow_period + self._signal_period
        if len(price_history) < min_len:
            return None

        fast_ema = ema(price_history, self._fast_period)
        slow_ema = ema(price_history, self._slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = ema(macd_line, self._signal_period)

        current = float(price_history[-1])
        diff = float(macd_line[-1] - signal_line[-1])
        if current == 0:
            return 0.0
        # Bullish when MACD > signal. The bearish below-signal half lives
        # in MACDExit, not here. Normalize by price so the score is
        # comparable across tickers.
        return self.clamp(diff / current * 100, 0.0, 1.0)

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return (
            f"Bullish when MACD({self._fast_period},{self._slow_period}) "
            f"is above the {self._signal_period}-period signal line"
        )
