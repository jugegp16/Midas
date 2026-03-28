"""Stop loss strategy: sell when unrealized loss exceeds threshold."""

from __future__ import annotations

import numpy as np
import pandas as pd

from midas.models import AssetSuitability, StrategyTier
from midas.strategies.base import Strategy


class StopLoss(Strategy):
    def __init__(self, loss_threshold: float = 0.10) -> None:
        self._loss_threshold = loss_threshold

    @property
    def tier(self) -> StrategyTier:
        return StrategyTier.PROTECTIVE

    def score(
        self,
        ticker: str,
        price_history: pd.Series,
        *,
        cost_basis: float | None = None,
        **kwargs: object,
    ) -> float | None:
        if cost_basis is None or cost_basis <= 0:
            return None

        current = float(np.asarray(price_history)[-1])
        loss = (cost_basis - current) / cost_basis

        if loss >= self._loss_threshold:
            return -self._clamp(
                (loss - self._loss_threshold) / self._loss_threshold
            )
        return 0.0

    @property
    def name(self) -> str:
        return f"StopLoss(loss_threshold={self._loss_threshold})"

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return (
            f"Sell when unrealized loss exceeds {self._loss_threshold:.0%} "
            f"of cost basis"
        )
