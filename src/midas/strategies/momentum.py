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
        # Momentum is buy-only: bullish when price is above MA, neutral when
        # below. Bearish below-MA signals are left to dedicated strategies
        # (MeanReversion, RSIOverbought) to avoid double-counting.
        if pct_from_ma <= 0:
            return 0.0
        return self._clamp(pct_from_ma / 0.05)

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return f"Bullish when price is above the {self._window}-day moving average"
