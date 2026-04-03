"""RSI overbought strategy: sell when RSI exceeds threshold."""

from __future__ import annotations

import numpy as np

from midas.models import AssetSuitability
from midas.strategies.base import Strategy


class RSIOverbought(Strategy):
    def __init__(self, window: int = 14, overbought_threshold: float = 70.0) -> None:
        self._window = window
        self._overbought_threshold = overbought_threshold

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

        # Continuous signal: RSI 50 = neutral (0), higher = bearish, lower = bullish.
        # Scale so that hitting the overbought threshold maps to -1.
        midpoint = 50.0
        distance = rsi - midpoint  # positive when RSI > 50 (bearish)
        scale = self._overbought_threshold - midpoint  # e.g. 70 - 50 = 20
        if scale == 0:
            return 0.0
        return self._clamp(-distance / scale, -1.0, 1.0)

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return (
            f"Bearish above / bullish below RSI 50"
            f" ({self._window}-period, overbought at {self._overbought_threshold:.0f})"
        )
