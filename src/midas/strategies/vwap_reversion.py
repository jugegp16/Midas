"""VWAP reversion strategy: buy below average price, sell above.

Note: Without volume data, this uses a simple moving average as a proxy
for VWAP. With volume data available via the provider, this could be
extended to use true volume-weighted average price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from midas.models import AssetSuitability, Direction, Signal
from midas.strategies.base import Strategy


class VWAPReversion(Strategy):
    def __init__(self, window: int = 20, threshold: float = 0.02) -> None:
        self._window = window
        self._threshold = threshold

    def evaluate(
        self,
        ticker: str,
        price_history: pd.Series,
        **kwargs: object,
    ) -> list[Signal]:
        if len(price_history) < self._window:
            return []

        values = np.asarray(price_history)
        current = float(values[-1])
        avg_price = float(values[-self._window :].mean())

        if avg_price == 0:
            return []

        deviation = (current - avg_price) / avg_price

        if deviation <= -self._threshold:
            return [self._make_signal(
                ticker,
                Direction.BUY,
                strength=abs(deviation) / (self._threshold * 3),
                reasoning=(
                    f"{ticker} at ${current:.2f} is {abs(deviation):.1%} below "
                    f"{self._window}-day VWAP proxy ${avg_price:.2f}"
                ),
                price=current,
            )]
        elif deviation >= self._threshold:
            return [self._make_signal(
                ticker,
                Direction.SELL,
                strength=deviation / (self._threshold * 3),
                reasoning=(
                    f"{ticker} at ${current:.2f} is {deviation:.1%} above "
                    f"{self._window}-day VWAP proxy ${avg_price:.2f}"
                ),
                price=current,
            )]
        return []

    @property
    def name(self) -> str:
        return f"VWAPReversion(window={self._window}, threshold={self._threshold})"

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.LARGE_CAP, AssetSuitability.BROAD_MARKET_ETF]

    @property
    def description(self) -> str:
        return (
            f"Buy below / sell above {self._window}-day average price "
            f"(VWAP proxy) by {self._threshold:.0%}"
        )
