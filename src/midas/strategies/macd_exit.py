"""MACD exit rule: sell on bearish MACD crossover (MACD line below signal).

Companion to ``MACDCrossover`` (the entry signal). The bearish crossover
is a technical signal that does not use cost basis or high-water mark.
"""

from __future__ import annotations

from midas.data.price_history import PriceHistory
from midas.models import AssetSuitability
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

    def clamp_target(
        self,
        ticker: str,
        proposed_target: float,
        price_history: PriceHistory,
        cost_basis: float,
        high_water_mark: float,
    ) -> float:
        if proposed_target <= 0:
            return proposed_target
        prices = price_history.close
        min_len = self._slow_period + self._signal_period
        if len(prices) < min_len:
            return proposed_target
        current = float(prices[-1])
        if current <= 0:
            return proposed_target

        fast_ema = ema(prices, self._fast_period)
        slow_ema = ema(prices, self._slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = ema(macd_line, self._signal_period)
        diff = float(macd_line[-1] - signal_line[-1])

        if diff < 0:
            return 0.0
        return proposed_target

    def clamp_reason(
        self,
        ticker: str,
        price_history: PriceHistory,
        cost_basis: float,
        high_water_mark: float,
    ) -> str:
        prices = price_history.close
        fast_ema = ema(prices, self._fast_period)
        slow_ema = ema(prices, self._slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = ema(macd_line, self._signal_period)
        diff = float(macd_line[-1] - signal_line[-1])
        return f"MACD below signal: MACD {macd_line[-1]:+.3f} vs signal {signal_line[-1]:+.3f} (diff {diff:+.3f})"

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return (
            f"Sell when MACD({self._fast_period},{self._slow_period}) "
            f"crosses below the {self._signal_period}-period signal line"
        )
