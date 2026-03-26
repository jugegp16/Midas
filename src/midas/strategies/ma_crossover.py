"""Moving average crossover: buy on golden cross, sell on death cross."""

from __future__ import annotations

import numpy as np
import pandas as pd

from midas.models import AssetSuitability, Direction, Signal
from midas.strategies.base import Strategy


class MovingAverageCrossover(Strategy):
    def __init__(self, short_window: int = 20, long_window: int = 50) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def evaluate(
        self,
        ticker: str,
        price_history: pd.Series,
        **kwargs: object,
    ) -> list[Signal]:
        if len(price_history) < self._long_window + 1:
            return []

        values = np.asarray(price_history)
        current = float(values[-1])

        short_ma = float(values[-self._short_window :].mean())
        long_ma = float(values[-self._long_window :].mean())
        prev_short_ma = float(values[-(self._short_window + 1) : -1].mean())
        prev_long_ma = float(values[-(self._long_window + 1) : -1].mean())

        if long_ma == 0:
            return []

        # Golden cross: short MA crosses above long MA
        if prev_short_ma <= prev_long_ma and short_ma > long_ma:
            spread = (short_ma - long_ma) / long_ma
            return [self._make_signal(
                ticker,
                Direction.BUY,
                strength=spread / 0.05,
                reasoning=(
                    f"{ticker} golden cross: {self._short_window}-day MA "
                    f"(${short_ma:.2f}) crossed above {self._long_window}-day "
                    f"MA (${long_ma:.2f}) at ${current:.2f}"
                ),
                price=current,
            )]

        # Death cross: short MA crosses below long MA
        if prev_short_ma >= prev_long_ma and short_ma < long_ma:
            spread = (long_ma - short_ma) / long_ma
            return [self._make_signal(
                ticker,
                Direction.SELL,
                strength=spread / 0.05,
                reasoning=(
                    f"{ticker} death cross: {self._short_window}-day MA "
                    f"(${short_ma:.2f}) crossed below {self._long_window}-day "
                    f"MA (${long_ma:.2f}) at ${current:.2f}"
                ),
                price=current,
            )]

        return []

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
