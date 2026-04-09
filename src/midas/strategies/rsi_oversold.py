"""RSI oversold strategy: buy when RSI drops below threshold."""

from __future__ import annotations

import numpy as np

from midas.models import AssetSuitability
from midas.strategies.base import Strategy


class RSIOversold(Strategy):
    def __init__(self, window: int = 14, oversold_threshold: float = 30.0) -> None:
        self._window = window
        self._oversold_threshold = oversold_threshold

    @property
    def warmup_period(self) -> int:
        # Both score() and precompute() use SMA (not Wilder EMA), so the true
        # minimum is window + 1 bars (window deltas). No stability multiplier needed.
        return self._window + 1

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
        valid = avg_loss != 0
        rs = np.where(valid, avg_gain / np.where(valid, avg_loss, 1.0), 0.0)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        midpoint = 50.0
        scale = midpoint - self._oversold_threshold
        if scale != 0:
            below_mid = valid & (rsi < midpoint)
            result[below_mid] = np.clip((midpoint - rsi[below_mid]) / scale, 0.0, 1.0)
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

        if avg_loss == 0:
            return 0.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # RSIOversold is buy-only: bullish when RSI < 50, neutral when above.
        # The bearish above-50 signal is RSIOverbought's responsibility —
        # keeping them one-sided avoids double-counting when both are active.
        midpoint = 50.0
        if rsi >= midpoint:
            return 0.0
        distance = midpoint - rsi  # positive when RSI < 50
        scale = midpoint - self._oversold_threshold  # e.g. 50 - 30 = 20
        if scale == 0:
            return 0.0
        return self.clamp(distance / scale, 0.0, 1.0)

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return f"Bullish when RSI is below 50 ({self._window}-period, oversold at {self._oversold_threshold:.0f})"
