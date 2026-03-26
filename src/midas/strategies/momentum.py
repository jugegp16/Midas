"""Momentum strategy: buy when price crosses above moving average."""

from __future__ import annotations

import numpy as np
import pandas as pd

from midas.models import AssetSuitability, Direction, Signal
from midas.strategies.base import Strategy


class Momentum(Strategy):
    def __init__(self, window: int = 20) -> None:
        self._window = window

    def evaluate(
        self,
        ticker: str,
        price_history: pd.Series,
        **kwargs: object,
    ) -> list[Signal]:
        if len(price_history) < self._window + 1:
            return []

        values = np.asarray(price_history)
        current, prev = float(values[-1]), float(values[-2])
        ma = float(values[-self._window:].mean())
        prev_ma = float(values[-(self._window + 1):-1].mean())

        if prev <= prev_ma and current > ma:
            pct_above = (current - ma) / ma
            return [self._make_signal(
                ticker,
                Direction.BUY,
                strength=pct_above / 0.05,
                reasoning=(
                    f"{ticker} crossed above {self._window}-day avg "
                    f"(${ma:.2f}) at ${current:.2f}"
                ),
                price=current,
            )]
        return []

    @property
    def name(self) -> str:
        return f"Momentum(window={self._window})"

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return (
            f"Buy when price crosses above the {self._window}-day "
            f"moving average from below"
        )
