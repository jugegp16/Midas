"""RSI oversold strategy: buy when RSI drops below threshold."""

from __future__ import annotations

import numpy as np
import pandas as pd

from midas.models import AssetSuitability
from midas.strategies.base import Strategy


class RSIOversold(Strategy):
    def __init__(self, window: int = 14, oversold_threshold: float = 30.0) -> None:
        self._window = window
        self._oversold_threshold = oversold_threshold

    def score(
        self,
        ticker: str,
        price_history: pd.Series,
        **kwargs: object,
    ) -> float | None:
        if len(price_history) < self._window + 1:
            return None

        values = np.asarray(price_history)
        deltas = np.diff(values[-(self._window + 1) :])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = float(gains.mean())
        avg_loss = float(losses.mean())

        if avg_loss == 0:
            return 0.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        if rsi <= self._oversold_threshold:
            return self._clamp(
                (self._oversold_threshold - rsi) / self._oversold_threshold
            )
        return 0.0

    @property
    def name(self) -> str:
        return (
            f"RSIOversold(window={self._window}, "
            f"oversold_threshold={self._oversold_threshold})"
        )

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return (
            f"Buy when {self._window}-period RSI drops below "
            f"{self._oversold_threshold:.0f}"
        )
