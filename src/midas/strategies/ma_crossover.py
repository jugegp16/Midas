"""Moving average crossover entry: buy when short MA > long MA (golden cross).

The bearish death cross half lives in the companion ``MovingAverageCrossoverExit``
strategy — entries and exits are separate types and never blended together.
"""

from __future__ import annotations

import numpy as np

from midas.data.price_history import PriceHistory
from midas.models import AssetSuitability
from midas.strategies.base import EntrySignal


class MovingAverageCrossover(EntrySignal):
    def __init__(self, short_window: int = 20, long_window: int = 50, spread_scale: float = 0.05) -> None:
        self._short_window = short_window
        self._long_window = long_window
        self._spread_scale = spread_scale

    @property
    def warmup_period(self) -> int:
        return self._long_window

    def precompute(self, price_history: PriceHistory) -> np.ndarray | None:
        prices = price_history.close
        num_bars = len(prices)
        long_win = self._long_window
        short_win = self._short_window
        scores = np.full(num_bars, np.nan)
        if num_bars < long_win:
            return scores
        cs = np.empty(num_bars + 1)
        cs[0] = 0.0
        np.cumsum(prices, out=cs[1:])
        short_rolling = (cs[short_win:] - cs[:-short_win]) / short_win
        long_rolling = (cs[long_win:] - cs[:-long_win]) / long_win
        short_at_day = short_rolling[long_win - short_win : num_bars - short_win + 1]
        long_at_day = long_rolling
        spread = np.where(long_at_day != 0, (short_at_day - long_at_day) / long_at_day, 0.0)
        scores[long_win - 1 :] = np.clip(spread / self._spread_scale, 0.0, 1.0)
        return scores

    def score(
        self,
        price_history: PriceHistory,
        **kwargs: object,
    ) -> float | None:
        prices = price_history.close
        if len(prices) < self._long_window:
            return None

        short_ma = float(prices[-self._short_window :].mean())
        long_ma = float(prices[-self._long_window :].mean())

        if long_ma == 0:
            return 0.0

        # Bullish (golden cross) when short MA > long MA. Bearish (death
        # cross) is handled by MovingAverageCrossoverExit, not here.
        spread = (short_ma - long_ma) / long_ma
        return self.clamp(spread / self._spread_scale, 0.0, 1.0)

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return f"Bullish when {self._short_window}-day MA > {self._long_window}-day MA (golden cross)"
