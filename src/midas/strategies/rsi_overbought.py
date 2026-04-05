"""RSI overbought strategy: sell when RSI exceeds threshold."""

from __future__ import annotations

import numpy as np

from midas.models import AssetSuitability
from midas.strategies.base import Strategy


class RSIOverbought(Strategy):
    def __init__(self, window: int = 14, overbought_threshold: float = 70.0) -> None:
        self._window = window
        self._overbought_threshold = overbought_threshold

    def precompute(self, prices: np.ndarray) -> np.ndarray | None:
        n = len(prices)
        w = self._window
        scores = np.full(n, np.nan)
        if n < w + 1:
            return scores
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        g_cs = np.empty(len(gains) + 1)
        g_cs[0] = 0.0
        np.cumsum(gains, out=g_cs[1:])
        l_cs = np.empty(len(losses) + 1)
        l_cs[0] = 0.0
        np.cumsum(losses, out=l_cs[1:])
        avg_gain = (g_cs[w:] - g_cs[:-w]) / w
        avg_loss = (l_cs[w:] - l_cs[:-w]) / w
        result = np.zeros(len(avg_gain))
        both_zero = (avg_gain == 0) & (avg_loss == 0)
        loss_zero = (avg_loss == 0) & ~both_zero
        normal = ~both_zero & ~loss_zero
        rsi = np.zeros(len(avg_gain))
        rsi[loss_zero] = 100.0
        rs = np.where(normal, avg_gain / np.where(normal, avg_loss, 1.0), 0.0)
        rsi[normal] = 100.0 - 100.0 / (1.0 + rs[normal])
        midpoint = 50.0
        scale = self._overbought_threshold - midpoint
        if scale != 0:
            above_mid = rsi > midpoint
            result[above_mid] = -np.clip((rsi[above_mid] - midpoint) / scale, 0.0, 1.0)
        scores[w:] = result
        return scores

    def score(
        self,
        price_history: np.ndarray,
        **kwargs: object,
    ) -> float | None:
        if len(price_history) < self._window + 1:
            return None

        deltas = np.diff(price_history[-(self._window + 1) :])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = float(gains.mean())
        avg_loss = float(losses.mean())

        if avg_gain == 0 and avg_loss == 0:
            return 0.0
        elif avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        # RSIOverbought is sell-only: bearish when RSI > 50, neutral when below.
        # The bullish below-50 signal is RSIOversold's responsibility —
        # keeping them one-sided avoids double-counting when both are active.
        midpoint = 50.0
        if rsi <= midpoint:
            return 0.0
        distance = rsi - midpoint  # positive when RSI > 50
        scale = self._overbought_threshold - midpoint  # e.g. 70 - 50 = 20
        if scale == 0:
            return 0.0
        return -self.clamp(distance / scale, 0.0, 1.0)

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return f"Bearish when RSI is above 50 ({self._window}-period, overbought at {self._overbought_threshold:.0f})"
