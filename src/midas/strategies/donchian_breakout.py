"""Donchian breakout: buy when price breaks above the rolling N-bar high."""

from __future__ import annotations

import numpy as np

from midas.data.price_history import PriceHistory
from midas.models import AssetSuitability
from midas.strategies.base import EntrySignal


class DonchianBreakout(EntrySignal):
    def __init__(self, window: int = 20, breakout_scale: float = 0.02) -> None:
        self._window = window
        self._breakout_scale = breakout_scale

    @property
    def warmup_period(self) -> int:
        return self._window + 1

    def precompute(self, price_history: PriceHistory) -> np.ndarray | None:
        highs = price_history.high
        close = price_history.close
        n = len(close)
        w = self._window
        scores = np.full(n, np.nan)
        if n < w + 1:
            return scores
        windows = np.lib.stride_tricks.sliding_window_view(highs[:-1], w)
        prior_highs = windows.max(axis=1)
        current = close[w:]
        with np.errstate(divide="ignore", invalid="ignore"):
            excess_pct = np.where(prior_highs > 0, (current - prior_highs) / prior_highs, 0.0)
        raw = np.where(current > prior_highs, excess_pct / self._breakout_scale, 0.0)
        scores[w:] = np.clip(raw, 0.0, 1.0)
        return scores

    def score(
        self,
        price_history: PriceHistory,
        **kwargs: object,
    ) -> float | None:
        close = price_history.close
        if len(close) < self._window + 1:
            return None

        prior_high = float(price_history.high[-self._window - 1 : -1].max())
        current = float(close[-1])

        if prior_high <= 0 or current <= prior_high:
            return 0.0

        excess_pct = (current - prior_high) / prior_high
        return self.clamp(excess_pct / self._breakout_scale, 0.0, 1.0)

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return (
            f"Bullish when current price breaks above the highest high of "
            f"the prior {self._window} bars (Turtle-style breakout)"
        )
