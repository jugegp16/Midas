"""VWAP reversion strategy: buy below average price, sell above.

Note: Without volume data, this uses a simple moving average as a proxy
for VWAP. With volume data available via the provider, this could be
extended to use true volume-weighted average price.
"""

from __future__ import annotations

import numpy as np

from midas.models import AssetSuitability
from midas.strategies.base import Strategy


class VWAPReversion(Strategy):
    def __init__(self, window: int = 20, threshold: float = 0.02) -> None:
        self._window = window
        self._threshold = threshold

    @property
    def warmup_period(self) -> int:
        return self._window

    def precompute(self, prices: np.ndarray) -> np.ndarray | None:
        n = len(prices)
        w = self._window
        scores = np.full(n, np.nan)
        if n < w:
            return scores
        cs = np.empty(n + 1)
        cs[0] = 0.0
        np.cumsum(prices, out=cs[1:])
        avg = (cs[w:] - cs[:-w]) / w
        current = prices[w - 1 :]
        deviation = np.where(avg != 0, (current - avg) / avg, 0.0)
        scores[w - 1 :] = np.clip(-deviation / self._threshold, -1.0, 1.0)
        return scores

    def score(
        self,
        price_history: np.ndarray,
        **kwargs: object,
    ) -> float | None:
        if len(price_history) < self._window:
            return None

        current = float(price_history[-1])
        avg_price = float(price_history[-self._window :].mean())

        if avg_price == 0:
            return 0.0

        deviation = (current - avg_price) / avg_price

        # Continuous: negative deviation (below avg) = bullish, positive = bearish.
        # Scaled so that ±threshold maps to ∓1.
        return self.clamp(-deviation / self._threshold, -1.0, 1.0)

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.LARGE_CAP, AssetSuitability.BROAD_MARKET_ETF]

    @property
    def description(self) -> str:
        return f"Buy below / sell above {self._window}-day average price (VWAP proxy) by {self._threshold:.0%}"
