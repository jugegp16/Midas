"""Momentum strategy: buy when price crosses above moving average."""

from __future__ import annotations

import numpy as np

from midas.models import AssetSuitability
from midas.strategies.base import EntrySignal


class Momentum(EntrySignal):
    def __init__(self, window: int = 20, momentum_scale: float = 0.05) -> None:
        self._window = window
        self._momentum_scale = momentum_scale

    @property
    def warmup_period(self) -> int:
        return self._window

    def precompute(self, prices: np.ndarray) -> np.ndarray | None:
        n = len(prices)
        w = self._window
        scores = np.full(n, np.nan)
        if n < w:
            return scores
        cs = np.empty(n + 1)
        cs[0] = 0.0
        np.cumsum(prices, out=cs[1:])
        ma = (cs[w:] - cs[:-w]) / w
        current = prices[w - 1 :]
        pct_from_ma = np.where(ma != 0, (current - ma) / ma, 0.0)
        raw = np.where(pct_from_ma > 0, pct_from_ma / self._momentum_scale, 0.0)
        scores[w - 1 :] = np.clip(raw, 0.0, 1.0)
        return scores

    def score(
        self,
        price_history: np.ndarray,
        **kwargs: object,
    ) -> float | None:
        if len(price_history) < self._window:
            return None

        current = float(price_history[-1])
        ma = float(price_history[-self._window :].mean())

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
