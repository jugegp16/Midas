"""Mean reversion strategy: buy when price drops below moving average."""

from __future__ import annotations

import numpy as np

from midas.models import AssetSuitability
from midas.strategies.base import EntrySignal


class MeanReversion(EntrySignal):
    def __init__(self, window: int = 30, threshold: float = 0.10) -> None:
        self._window = window
        self._threshold = threshold

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
        pct_below = np.where(ma != 0, (ma - current) / ma, 0.0)
        scores[w - 1 :] = np.clip(pct_below / self._threshold, 0.0, 1.0)
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

        # How far below the MA (positive = below = bullish for mean reversion).
        pct_below = (ma - current) / ma
        # Buy-only entry signal: ramps from 0 at the MA to 1 at threshold and
        # beyond. The bearish "above MA" half is dropped — exits are handled
        # by ExitRule strategies (StopLoss, TrailingStop, ProfitTaking), not
        # by sign-flipping entry scores.
        return self.clamp(pct_below / self._threshold, 0.0, 1.0)

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.BROAD_MARKET_ETF, AssetSuitability.LARGE_CAP]

    @property
    def description(self) -> str:
        return f"Bullish below the {self._window}-day MA (scaled by {self._threshold:.0%} threshold)"
