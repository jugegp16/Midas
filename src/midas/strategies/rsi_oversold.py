"""RSI oversold strategy: buy when RSI drops below threshold."""

from __future__ import annotations

import numpy as np
import pandas as pd

from midas.models import AssetSuitability, Direction, Signal
from midas.strategies.base import Strategy


class RSIOversold(Strategy):
    def __init__(self, window: int = 14, oversold_threshold: float = 30.0) -> None:
        self._window = window
        self._oversold_threshold = oversold_threshold

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

        if avg_loss == 0:
            return []

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        current = float(values[-1])

        if rsi <= self._oversold_threshold:
            return [self._make_signal(
                ticker,
                Direction.BUY,
                strength=(self._oversold_threshold - rsi) / self._oversold_threshold,
                reasoning=(
                    f"{ticker} RSI({self._window}) at {rsi:.1f}, "
                    f"below oversold threshold {self._oversold_threshold:.0f}"
                ),
                price=current,
            )]
        return []

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
