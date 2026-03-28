"""Moving average crossover: buy on golden cross, sell on death cross."""

from __future__ import annotations

import numpy as np
import pandas as pd

from midas.models import AssetSuitability
from midas.strategies.base import Strategy


class MovingAverageCrossover(Strategy):
    def __init__(self, short_window: int = 20, long_window: int = 50) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def score(
        self,
        ticker: str,
        price_history: pd.Series,
        **kwargs: object,
    ) -> float | None:
        if len(price_history) < self._long_window + 1:
            return None

        values = np.asarray(price_history)

        short_ma = float(values[-self._short_window :].mean())
        long_ma = float(values[-self._long_window :].mean())
        prev_short_ma = float(values[-(self._short_window + 1) : -1].mean())
        prev_long_ma = float(values[-(self._long_window + 1) : -1].mean())

        if long_ma == 0:
            return 0.0

        # Golden cross -> bullish
        if prev_short_ma <= prev_long_ma and short_ma > long_ma:
            spread = (short_ma - long_ma) / long_ma
            return self._clamp(spread / 0.05)

        # Death cross -> bearish
        if prev_short_ma >= prev_long_ma and short_ma < long_ma:
            spread = (long_ma - short_ma) / long_ma
            return -self._clamp(spread / 0.05)

        return 0.0

    @property
    def name(self) -> str:
        return (
            f"MovingAverageCrossover(short={self._short_window}, "
            f"long={self._long_window})"
        )

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return (
            f"Buy on golden cross ({self._short_window}/{self._long_window}-day MA), "
            f"sell on death cross"
        )
