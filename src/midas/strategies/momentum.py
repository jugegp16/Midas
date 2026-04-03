"""Momentum strategy: buy when price crosses above moving average."""

from __future__ import annotations

import numpy as np

from midas.models import AssetSuitability
from midas.strategies.base import Strategy


class Momentum(Strategy):
    def __init__(self, window: int = 20) -> None:
        self._window = window

    def score(
        self,
        price_history: np.ndarray,
        **kwargs: object,
    ) -> float | None:
        if len(price_history) < self._window:
            return None

        current = float(price_history[-1])
        ma = float(price_history[-self._window :].mean())

        if ma == 0:
            return 0.0

        pct_from_ma = (current - ma) / ma
        # Positive when price is above MA, negative when below.
        # Scaled so that 5% above/below maps to ±1.
        return self._clamp(pct_from_ma / 0.05, -1.0, 1.0)

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return f"Bullish above / bearish below the {self._window}-day moving average"
