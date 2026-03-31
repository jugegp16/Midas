"""Dollar-cost averaging strategy: buy on a regular cadence."""

from __future__ import annotations

import pandas as pd

from midas.models import (
    AssetSuitability,
    Direction,
    MechanicalIntent,
    StrategyTier,
)
from midas.strategies.base import Strategy


class DollarCostAveraging(Strategy):
    def __init__(self, frequency_days: int = 14, amount: float = 500.0) -> None:
        self._frequency_days = frequency_days
        self._amount = amount

    @property
    def tier(self) -> StrategyTier:
        return StrategyTier.MECHANICAL

    def score(
        self,
        price_history: pd.Series,
        **kwargs: object,
    ) -> float | None:
        # MECHANICAL strategies don't participate in scoring.
        return None

    def generate_intents(
        self,
        ticker: str,
        price_history: pd.Series,
        **kwargs: object,
    ) -> list[MechanicalIntent]:
        if len(price_history) < self._frequency_days:
            return []

        if len(price_history) % self._frequency_days == 0:
            current = float(price_history.iloc[-1])
            return [
                MechanicalIntent(
                    ticker=ticker,
                    direction=Direction.BUY,
                    target_value=self._amount,
                    reason=(f"{ticker} DCA trigger: {self._frequency_days}-day interval reached at ${current:.2f}"),
                    source=self.name,
                )
            ]
        return []

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.BROAD_MARKET_ETF, AssetSuitability.LARGE_CAP]

    @property
    def description(self) -> str:
        return f"Buy every {self._frequency_days} trading days regardless of price"
