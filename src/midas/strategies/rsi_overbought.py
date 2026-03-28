"""RSI overbought strategy: sell when RSI exceeds threshold."""

from __future__ import annotations

import numpy as np
import pandas as pd

from midas.models import AssetSuitability
from midas.strategies.base import Strategy


class RSIOverbought(Strategy):
    def __init__(self, window: int = 14, overbought_threshold: float = 70.0) -> None:
        self._window = window
        self._overbought_threshold = overbought_threshold

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

        if avg_gain == 0 and avg_loss == 0:
            return 0.0
        elif avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        if rsi >= self._overbought_threshold:
            return -self._clamp(
                (rsi - self._overbought_threshold)
                / (100.0 - self._overbought_threshold)
            )
        return 0.0

    @property
    def name(self) -> str:
        return (
            f"RSIOverbought(window={self._window}, "
            f"overbought_threshold={self._overbought_threshold})"
        )

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return (
            f"Sell when {self._window}-period RSI exceeds "
            f"{self._overbought_threshold:.0f}"
        )
