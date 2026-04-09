"""Bollinger Band strategy: buy when price touches lower band."""

from __future__ import annotations

import numpy as np

from midas.models import AssetSuitability
from midas.strategies.base import EntrySignal


class BollingerBand(EntrySignal):
    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        self._window = window
        self._num_std = num_std

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
        sq_cs = np.empty(n + 1)
        sq_cs[0] = 0.0
        np.cumsum(prices**2, out=sq_cs[1:])
        rolling_sum = cs[w:] - cs[:-w]
        rolling_sq = sq_cs[w:] - sq_cs[:-w]
        ma = rolling_sum / w
        variance = (rolling_sq / w - ma**2) * w / (w - 1)
        std = np.sqrt(np.maximum(variance, 0.0))
        current = prices[w - 1 :]
        z = np.where(std != 0, (current - ma) / std, 0.0)
        scores[w - 1 :] = np.clip(-z / self._num_std, 0.0, 1.0)
        return scores

    def score(
        self,
        price_history: np.ndarray,
        **kwargs: object,
    ) -> float | None:
        if len(price_history) < self._window:
            return None

        window_data = price_history[-self._window :]
        ma = float(window_data.mean())
        std = float(window_data.std(ddof=1))

        if std == 0:
            return 0.0

        current = float(price_history[-1])
        # Z-score: how many std devs from the mean. Buy-only entry signal:
        # negative z (below MA) ramps from 0 at the MA to 1 at the lower band.
        # The bearish "above upper band" half is dropped — exits are handled
        # by ExitRule strategies.
        z = (current - ma) / std
        return self.clamp(-z / self._num_std, 0.0, 1.0)

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.BROAD_MARKET_ETF, AssetSuitability.LARGE_CAP]

    @property
    def description(self) -> str:
        return f"Bullish at the lower {self._num_std}-sigma Bollinger band of the {self._window}-day MA"
