"""Dollar-cost averaging strategy: buy on a regular cadence."""

from __future__ import annotations

import numpy as np

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
        self._last_trigger_len: int = 0

    @property
    def tier(self) -> StrategyTier:
        return StrategyTier.MECHANICAL

    def score(
        self,
        price_history: np.ndarray,
        **kwargs: object,
    ) -> float | None:
        # MECHANICAL strategies don't participate in scoring.
        return None

    def generate_intents(
        self,
        ticker: str,
        price_history: np.ndarray,
        **kwargs: object,
    ) -> list[MechanicalIntent]:
        n = len(price_history)
        if n < self._frequency_days:
            return []

        if self._last_trigger_len == 0:
            # Align the first trigger to the end of the first full window so
            # behaviour is deterministic regardless of warmup buffer size.
            self._last_trigger_len = n - (n % self._frequency_days or self._frequency_days)

        if n - self._last_trigger_len >= self._frequency_days:
            self._last_trigger_len = n
            current = float(price_history[-1])
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
