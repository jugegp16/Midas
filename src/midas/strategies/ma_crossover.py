"""Moving average crossover: buy on golden cross, sell on death cross."""

from __future__ import annotations

import numpy as np

from midas.models import AssetSuitability
from midas.strategies.base import Strategy


class MovingAverageCrossover(Strategy):
    def __init__(self, short_window: int = 20, long_window: int = 50, spread_scale: float = 0.05) -> None:
        self._short_window = short_window
        self._long_window = long_window
        self._spread_scale = spread_scale

    @property
    def warmup_period(self) -> int:
        return self._long_window

    def precompute(self, prices: np.ndarray) -> np.ndarray | None:
        n = len(prices)
        lw = self._long_window
        sw = self._short_window
        scores = np.full(n, np.nan)
        if n < lw:
            return scores
        cs = np.empty(n + 1)
        cs[0] = 0.0
        np.cumsum(prices, out=cs[1:])
        short_rolling = (cs[sw:] - cs[:-sw]) / sw
        long_rolling = (cs[lw:] - cs[:-lw]) / lw
        short_at_day = short_rolling[lw - sw : n - sw + 1]
        long_at_day = long_rolling
        spread = np.where(long_at_day != 0, (short_at_day - long_at_day) / long_at_day, 0.0)
        scores[lw - 1 :] = np.clip(spread / self._spread_scale, -1.0, 1.0)
        return scores

    def score(
        self,
        price_history: np.ndarray,
        **kwargs: object,
    ) -> float | None:
        if len(price_history) < self._long_window:
            return None

        short_ma = float(price_history[-self._short_window :].mean())
        long_ma = float(price_history[-self._long_window :].mean())

        if long_ma == 0:
            return 0.0

        # Positive when short MA > long MA (bullish), negative when below.
        spread = (short_ma - long_ma) / long_ma
        return self.clamp(spread / self._spread_scale, -1.0, 1.0)

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return f"Bullish when {self._short_window}-day MA > {self._long_window}-day MA, bearish when below"
