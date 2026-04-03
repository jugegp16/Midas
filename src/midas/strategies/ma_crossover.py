"""Moving average crossover: buy on golden cross, sell on death cross."""

from __future__ import annotations

import numpy as np

from midas.models import AssetSuitability
from midas.strategies.base import Strategy


class MovingAverageCrossover(Strategy):
    def __init__(self, short_window: int = 20, long_window: int = 50) -> None:
        self._short_window = short_window
        self._long_window = long_window

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
        return self._clamp(spread / 0.05, -1.0, 1.0)

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return f"Bullish when {self._short_window}-day MA > {self._long_window}-day MA, bearish when below"
