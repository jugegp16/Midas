"""Bollinger Band strategy: buy when price touches lower band."""

from __future__ import annotations

import numpy as np
import pandas as pd

from midas.models import AssetSuitability
from midas.strategies.base import Strategy


class BollingerBand(Strategy):
    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        self._window = window
        self._num_std = num_std

    def score(
        self,
        price_history: pd.Series,
        **kwargs: object,
    ) -> float | None:
        if len(price_history) < self._window:
            return None

        values = np.asarray(price_history)
        window_data = values[-self._window :]
        ma = float(window_data.mean())
        std = float(window_data.std(ddof=1))

        if std == 0:
            return 0.0

        lower_band = ma - self._num_std * std
        current = float(values[-1])

        if current <= lower_band:
            pct_below = (lower_band - current) / std
            return self._clamp(pct_below / self._num_std)
        return 0.0

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.BROAD_MARKET_ETF, AssetSuitability.LARGE_CAP]

    @property
    def description(self) -> str:
        return f"Buy when price touches the lower Bollinger Band ({self._window}-day MA - {self._num_std} std dev)"
