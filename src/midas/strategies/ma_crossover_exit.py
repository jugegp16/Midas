"""Moving average crossover exit: sell on death cross (short MA < long MA).

Companion to ``MovingAverageCrossover`` (the entry signal). The death cross
is a technical signal that does not use cost basis or high-water mark.
"""

from __future__ import annotations

from midas.data.price_history import PriceHistory
from midas.models import AssetSuitability
from midas.strategies.base import ExitRule


class MovingAverageCrossoverExit(ExitRule):
    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 50,
    ) -> None:
        self._short_window = short_window
        self._long_window = long_window

    @property
    def warmup_period(self) -> int:
        return self._long_window

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
        if len(prices) < self._long_window:
            return proposed_target
        current = float(prices[-1])
        if current <= 0:
            return proposed_target

        short_ma = float(prices[-self._short_window :].mean())
        long_ma = float(prices[-self._long_window :].mean())
        if long_ma == 0:
            return proposed_target

        spread = (short_ma - long_ma) / long_ma
        if spread < 0:
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
        short_ma = float(prices[-self._short_window :].mean())
        long_ma = float(prices[-self._long_window :].mean())
        spread = (short_ma - long_ma) / long_ma if long_ma else 0.0
        return (
            f"Death cross: {self._short_window}-day MA ${short_ma:.2f} "
            f"below {self._long_window}-day MA ${long_ma:.2f} ({spread:.1%})"
        )

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return f"Sell on death cross ({self._short_window}-day MA below {self._long_window}-day MA)"
