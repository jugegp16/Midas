"""RSI overbought strategy: sell when RSI exceeds threshold."""

from __future__ import annotations

import numpy as np
import pandas as pd

from midas.models import AssetSuitability, Direction, Signal
from midas.strategies.base import Strategy


class RSIOverbought(Strategy):
    def __init__(self, window: int = 14, overbought_threshold: float = 70.0) -> None:
        self._window = window
        self._overbought_threshold = overbought_threshold

    def evaluate(
        self,
        ticker: str,
        price_history: pd.Series,
        **kwargs: object,
    ) -> list[Signal]:
        if len(price_history) < self._window + 1:
            return []

        values = np.asarray(price_history)
        deltas = np.diff(values[-(self._window + 1) :])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = float(gains.mean())
        avg_loss = float(losses.mean())

        if avg_gain == 0 and avg_loss == 0:
            return []  # No price movement — RSI undefined
        elif avg_loss == 0:
            rsi = 100.0  # All gains, no losses
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        current = float(values[-1])

        if rsi >= self._overbought_threshold:
            return [self._make_signal(
                ticker,
                Direction.SELL,
                strength=(rsi - self._overbought_threshold)
                / (100.0 - self._overbought_threshold),
                reasoning=(
                    f"{ticker} RSI({self._window}) at {rsi:.1f}, "
                    f"above overbought threshold {self._overbought_threshold:.0f}"
                ),
                price=current,
            )]
        return []

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
