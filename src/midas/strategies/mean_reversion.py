"""Mean reversion strategy: buy when price drops below moving average."""

from __future__ import annotations

import numpy as np
import pandas as pd

from midas.models import AssetSuitability
from midas.strategies.base import Strategy


class MeanReversion(Strategy):
    def __init__(self, window: int = 30, threshold: float = 0.10) -> None:
        self._window = window
        self._threshold = threshold

    def score(
        self,
        ticker: str,
        price_history: pd.Series,
        **kwargs: object,
    ) -> float | None:
        if len(price_history) < self._window:
            return None

        values = np.asarray(price_history)
        current = float(values[-1])
        ma = float(values[-self._window:].mean())

        if ma == 0:
            return 0.0

        pct_below = (ma - current) / ma
        if pct_below >= self._threshold:
            return self._clamp((pct_below - self._threshold) / self._threshold)
        return 0.0

    @property
    def name(self) -> str:
        return f"MeanReversion(window={self._window}, threshold={self._threshold})"

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.BROAD_MARKET_ETF, AssetSuitability.LARGE_CAP]

    @property
    def description(self) -> str:
        return (
            f"Buy when price is >{self._threshold:.0%} below "
            f"the {self._window}-day moving average"
        )
