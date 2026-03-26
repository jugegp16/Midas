"""Profit taking strategy: sell when unrealized gains exceed threshold."""

from __future__ import annotations

import numpy as np
import pandas as pd

from midas.models import AssetSuitability, Direction, Signal
from midas.strategies.base import Strategy


class ProfitTaking(Strategy):
    def __init__(self, gain_threshold: float = 0.20) -> None:
        self._gain_threshold = gain_threshold

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
        gain = (current - cost_basis) / cost_basis

        if gain >= self._gain_threshold:
            return [self._make_signal(
                ticker,
                Direction.SELL,
                strength=(gain - self._gain_threshold) / self._gain_threshold,
                reasoning=(
                    f"{ticker} up {gain:.0%} from "
                    f"cost basis ${cost_basis:.2f}"
                ),
                price=current,
            )]
        return []

    @property
    def name(self) -> str:
        return f"ProfitTaking(gain_threshold={self._gain_threshold})"

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return (
            f"Sell when unrealized gains exceed {self._gain_threshold:.0%} "
            f"of cost basis"
        )
