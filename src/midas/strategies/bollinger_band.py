"""Bollinger Band strategy: buy when price touches lower band."""

from __future__ import annotations

import numpy as np

from midas.models import AssetSuitability
from midas.strategies.base import Strategy


class BollingerBand(Strategy):
    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        self._window = window
        self._num_std = num_std

    def score(
        self,
        price_history: np.ndarray,
        **kwargs: object,
    ) -> float | None:
        if len(price_history) < self._window:
            return None

        window_data = price_history[-self._window :]
        ma = float(window_data.mean())
        std = float(window_data.std(ddof=1))

        if std == 0:
            return 0.0

        current = float(price_history[-1])
        # Z-score: how many std devs from the mean.
        # Negative z = below MA = bullish (buy the dip).
        # Positive z = above MA = bearish (stretched up).
        z = (current - ma) / std
        # Scale so that ±num_std maps to ∓1.
        return self.clamp(-z / self._num_std, -1.0, 1.0)

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.BROAD_MARKET_ETF, AssetSuitability.LARGE_CAP]

    @property
    def description(self) -> str:
        return f"Bullish below / bearish above the {self._window}-day MA, scaled by {self._num_std} std dev bands"
