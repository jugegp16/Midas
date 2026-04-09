"""Moving average crossover exit: sell on death cross (short MA < long MA).

Companion to ``MovingAverageCrossover`` (the entry signal). The two strategies
are deliberately split into separate types — entries and exits never blend
together. The death cross is technical-only and lot-unaware: when triggered,
all open shares of the ticker are sold regardless of cost basis.
"""

from __future__ import annotations

import numpy as np

from midas.models import AssetSuitability, ExitIntent, PositionLot
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

    def evaluate_exit(
        self,
        ticker: str,
        lots: list[PositionLot],
        price_history: np.ndarray,
    ) -> list[ExitIntent]:
        if not lots or len(price_history) < self._long_window:
            return []
        current = float(price_history[-1])
        if current <= 0:
            return []

        short_ma = float(price_history[-self._short_window :].mean())
        long_ma = float(price_history[-self._long_window :].mean())
        if long_ma == 0:
            return []

        spread = (short_ma - long_ma) / long_ma
        # Death cross: short MA below long MA.
        if spread >= 0:
            return []

        reason = (
            f"Death cross: {self._short_window}-day MA ${short_ma:.2f} "
            f"below {self._long_window}-day MA ${long_ma:.2f} ({spread:.1%})"
        )
        return self.sell_all(ticker, lots, current, reason)

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return f"Sell entire position on death cross ({self._short_window}-day MA below {self._long_window}-day MA)"
