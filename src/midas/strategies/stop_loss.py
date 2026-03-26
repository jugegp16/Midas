"""Stop loss strategy: sell when unrealized loss exceeds threshold."""

from __future__ import annotations

import numpy as np
import pandas as pd

from midas.models import AssetSuitability, Direction, Signal
from midas.strategies.base import Strategy


class StopLoss(Strategy):
    def __init__(self, loss_threshold: float = 0.10) -> None:
        self._loss_threshold = loss_threshold

    def evaluate(
        self,
        ticker: str,
        price_history: pd.Series,
        *,
        cost_basis: float | None = None,
        **kwargs: object,
    ) -> list[Signal]:
        if cost_basis is None or cost_basis <= 0:
            return []

        current = float(np.asarray(price_history)[-1])
        loss = (cost_basis - current) / cost_basis

        if loss >= self._loss_threshold:
            return [self._make_signal(
                ticker,
                Direction.SELL,
                strength=(loss - self._loss_threshold) / self._loss_threshold,
                reasoning=(
                    f"{ticker} down {loss:.0%} from "
                    f"cost basis ${cost_basis:.2f}"
                ),
                price=current,
            )]
        return []

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
