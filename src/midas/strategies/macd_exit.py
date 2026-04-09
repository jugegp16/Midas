"""MACD exit rule: sell on bearish MACD crossover (MACD line below signal).

Companion to ``MACDCrossover`` (the entry signal). The two strategies are
deliberately split into separate types — entries and exits never blend
together. The bearish crossover is technical-only and lot-unaware: when
triggered, all open shares of the ticker are sold regardless of cost basis.
"""

from __future__ import annotations

import numpy as np

from midas.models import AssetSuitability, ExitIntent, PositionLot
from midas.strategies.base import RECURSIVE_WARMUP_MULTIPLIER, ExitRule
from midas.strategies.macd_crossover import ema


class MACDExit(ExitRule):
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> None:
        self._fast_period = fast_period
        self._slow_period = slow_period
        self._signal_period = signal_period

    @property
    def warmup_period(self) -> int:
        return self._slow_period * RECURSIVE_WARMUP_MULTIPLIER + self._signal_period

    def evaluate_exit(
        self,
        ticker: str,
        lots: list[PositionLot],
        price_history: np.ndarray,
    ) -> list[ExitIntent]:
        min_len = self._slow_period + self._signal_period
        if not lots or len(price_history) < min_len:
            return []
        current = float(price_history[-1])
        if current <= 0:
            return []

        fast_ema = ema(price_history, self._fast_period)
        slow_ema = ema(price_history, self._slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = ema(macd_line, self._signal_period)
        diff = float(macd_line[-1] - signal_line[-1])

        # Bearish: MACD below signal line.
        if diff >= 0:
            return []

        reason = f"MACD below signal: MACD {macd_line[-1]:+.3f} vs signal {signal_line[-1]:+.3f} (diff {diff:+.3f})"
        return self.sell_all(ticker, lots, current, reason)

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return (
            f"Sell entire position when MACD({self._fast_period},{self._slow_period}) "
            f"crosses below the {self._signal_period}-period signal line"
        )
