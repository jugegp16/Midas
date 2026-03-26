"""MACD crossover strategy: buy when MACD line crosses above signal line."""

from __future__ import annotations

import numpy as np
import pandas as pd

from midas.models import AssetSuitability, Direction, Signal
from midas.strategies.base import Strategy


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Compute exponential moving average."""
    alpha = 2.0 / (period + 1)
    result = np.empty_like(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


class MACDCrossover(Strategy):
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> None:
        self._fast_period = fast_period
        self._slow_period = slow_period
        self._signal_period = signal_period

    def evaluate(
        self,
        ticker: str,
        price_history: pd.Series,
        **kwargs: object,
    ) -> list[Signal]:
        min_len = self._slow_period + self._signal_period
        if len(price_history) < min_len:
            return []

        values = np.asarray(price_history, dtype=float)
        fast_ema = _ema(values, self._fast_period)
        slow_ema = _ema(values, self._slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = _ema(macd_line, self._signal_period)

        # Check for crossover: MACD was below signal, now above
        if macd_line[-2] <= signal_line[-2] and macd_line[-1] > signal_line[-1]:
            current = float(values[-1])
            diff = float(macd_line[-1] - signal_line[-1])
            return [self._make_signal(
                ticker,
                Direction.BUY,
                strength=min(abs(diff) / current * 100, 1.0),
                reasoning=(
                    f"{ticker} MACD({self._fast_period},{self._slow_period},"
                    f"{self._signal_period}) crossed above signal line "
                    f"at ${current:.2f}"
                ),
                price=current,
            )]
        return []

    @property
    def name(self) -> str:
        return (
            f"MACDCrossover(fast={self._fast_period}, "
            f"slow={self._slow_period}, signal={self._signal_period})"
        )

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return (
            f"Buy when MACD({self._fast_period},{self._slow_period}) "
            f"crosses above {self._signal_period}-period signal line"
        )
