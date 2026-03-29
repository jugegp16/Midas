"""VWAP reversion strategy: buy below average price, sell above.

Note: Without volume data, this uses a simple moving average as a proxy
for VWAP. With volume data available via the provider, this could be
extended to use true volume-weighted average price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from midas.models import AssetSuitability
from midas.strategies.base import Strategy


class VWAPReversion(Strategy):
    def __init__(self, window: int = 20, threshold: float = 0.02) -> None:
        self._window = window
        self._threshold = threshold

    def score(
        self,
        price_history: pd.Series,
        **kwargs: object,
    ) -> float | None:
        if len(price_history) < self._window:
            return None

        values = np.asarray(price_history)
        current = float(values[-1])
        avg_price = float(values[-self._window :].mean())

        if avg_price == 0:
            return 0.0

        deviation = (current - avg_price) / avg_price

        if deviation <= -self._threshold:
            # Below VWAP -> bullish
            return self._clamp(abs(deviation) / (self._threshold * 3))
        elif deviation >= self._threshold:
            # Above VWAP -> bearish
            return -self._clamp(deviation / (self._threshold * 3))
        return 0.0

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.LARGE_CAP, AssetSuitability.BROAD_MARKET_ETF]

    @property
    def description(self) -> str:
        return (
            f"Buy below / sell above {self._window}-day average price "
            f"(VWAP proxy) by {self._threshold:.0%}"
        )
