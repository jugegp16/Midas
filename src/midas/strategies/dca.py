"""Dollar-cost averaging strategy: buy on a regular cadence."""

from __future__ import annotations

import pandas as pd

from midas.models import AssetSuitability, Direction, Signal
from midas.strategies.base import Strategy


class DollarCostAveraging(Strategy):
    def __init__(self, frequency_days: int = 14) -> None:
        self._frequency_days = frequency_days

    def evaluate(
        self,
        ticker: str,
        price_history: pd.Series,
        **kwargs: object,
    ) -> list[Signal]:
        if len(price_history) < self._frequency_days:
            return []

        # Signal fires when the number of trading days in the series
        # is a multiple of the frequency — i.e. "it's time to buy again."
        if len(price_history) % self._frequency_days == 0:
            current = float(price_history.iloc[-1])
            return [self._make_signal(
                ticker,
                Direction.BUY,
                strength=0.5,  # constant, disciplined buying
                reasoning=(
                    f"{ticker} DCA trigger: {self._frequency_days}-day "
                    f"interval reached at ${current:.2f}"
                ),
                price=current,
            )]
        return []

    @property
    def name(self) -> str:
        return f"DollarCostAveraging(frequency_days={self._frequency_days})"

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.BROAD_MARKET_ETF, AssetSuitability.LARGE_CAP]

    @property
    def description(self) -> str:
        return f"Buy every {self._frequency_days} trading days regardless of price"
