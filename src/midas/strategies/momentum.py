"""Momentum strategy: buy when price crosses above moving average."""

from __future__ import annotations

import numpy as np

from midas.data.price_history import PriceHistory
from midas.models import AssetSuitability
from midas.strategies.base import EntrySignal


class Momentum(EntrySignal):
    def __init__(self, window: int = 20, momentum_scale: float = 0.05) -> None:
        self._window = window
        self._momentum_scale = momentum_scale

    @property
    def warmup_period(self) -> int:
        return self._window

    def precompute(self, price_history: PriceHistory) -> np.ndarray | None:
        prices = price_history.close
        num_bars = len(prices)
        window = self._window
        scores = np.full(num_bars, np.nan)
        if num_bars < window:
            return scores
        cs = np.empty(num_bars + 1)
        cs[0] = 0.0
        np.cumsum(prices, out=cs[1:])
        ma = (cs[window:] - cs[:-window]) / window
        current = prices[window - 1 :]
        pct_from_ma = np.where(ma != 0, (current - ma) / ma, 0.0)
        raw = np.where(pct_from_ma > 0, pct_from_ma / self._momentum_scale, 0.0)
        scores[window - 1 :] = np.clip(raw, 0.0, 1.0)
        return scores

    def score(
        self,
        price_history: PriceHistory,
        **kwargs: object,
    ) -> float | None:
        prices = price_history.close
        if len(prices) < self._window:
            return None

        current = float(prices[-1])
        ma = float(prices[-self._window :].mean())

        if ma == 0:
            return 0.0

        pct_from_ma = (current - ma) / ma
        # Momentum is buy-only: bullish when price is above MA, neutral when
        # below. The entry-signal contract requires scores in [0, 1]; bearish
        # below-MA signals should fire via an ExitRule rather than producing a
        # negative score here.
        if pct_from_ma <= 0:
            return 0.0
        return self.clamp(pct_from_ma / self._momentum_scale, 0.0, 1.0)

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return f"Bullish when price is above the {self._window}-day moving average"
