"""VWAP reversion strategy: buy below average price, sell above.

Note: Without volume data, this uses a simple moving average as a proxy
for VWAP. With volume data available via the provider, this could be
extended to use true volume-weighted average price.
"""

from __future__ import annotations

import numpy as np

from midas.data.price_history import PriceHistory
from midas.models import AssetSuitability
from midas.strategies.base import EntrySignal


class VWAPReversion(EntrySignal):
    def __init__(self, window: int = 20, threshold: float = 0.02) -> None:
        self._window = window
        self._threshold = threshold

    @property
    def warmup_period(self) -> int:
        return self._window

    def precompute(self, price_history: PriceHistory) -> np.ndarray | None:
        prices = price_history.close
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
        scores[w - 1 :] = np.clip(-deviation / self._threshold, 0.0, 1.0)
        return scores

    def score(
        self,
        price_history: PriceHistory,
        **kwargs: object,
    ) -> float | None:
        prices = price_history.close
        if len(prices) < self._window:
            return None

        current = float(prices[-1])
        avg_price = float(prices[-self._window :].mean())

        if avg_price == 0:
            return 0.0

        deviation = (current - avg_price) / avg_price

        # Buy-only entry signal: negative deviation (below avg) ramps from 0
        # to 1. The bearish "above avg" half is dropped — exits are handled
        # by ExitRule strategies.
        return self.clamp(-deviation / self._threshold, 0.0, 1.0)

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.LARGE_CAP, AssetSuitability.BROAD_MARKET_ETF]

    @property
    def description(self) -> str:
        return f"Bullish below the {self._window}-day average price (VWAP proxy) by {self._threshold:.0%}"
