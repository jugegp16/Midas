"""Keltner Channel breakout: bullish when price breaks above SMA + k x ATR."""

from __future__ import annotations

import numpy as np

from midas.models import AssetSuitability
from midas.strategies.base import EntrySignal


class KeltnerChannel(EntrySignal):
    def __init__(self, window: int = 20, multiplier: float = 2.0) -> None:
        self._window = window
        self._multiplier = multiplier

    @property
    def warmup_period(self) -> int:
        return self._window

    def precompute(self, prices: np.ndarray) -> np.ndarray | None:
        n = len(prices)
        w = self._window
        scores = np.full(n, np.nan)
        if n < w or w < 2:
            return scores
        cs = np.empty(n + 1)
        cs[0] = 0.0
        np.cumsum(prices, out=cs[1:])
        centerlines = (cs[w:] - cs[:-w]) / w
        abs_diffs = np.abs(np.diff(prices))
        cs_diff = np.empty(n)
        cs_diff[0] = 0.0
        np.cumsum(abs_diffs, out=cs_diff[1:])
        atr_series = (cs_diff[w - 1 :] - cs_diff[: n - w + 1]) / (w - 1)
        upper = centerlines + self._multiplier * atr_series
        current = prices[w - 1 :]
        with np.errstate(divide="ignore", invalid="ignore"):
            excess_atr = np.where(atr_series > 0, (current - upper) / atr_series, 0.0)
        raw = np.where(current > upper, excess_atr, 0.0)
        scores[w - 1 :] = np.clip(raw, 0.0, 1.0)
        return scores

    def score(
        self,
        price_history: np.ndarray,
        **kwargs: object,
    ) -> float | None:
        if len(price_history) < self._window:
            return None

        recent = price_history[-self._window :]
        centerline = float(recent.mean())
        atr = float(np.abs(np.diff(recent)).mean())

        if atr <= 0:
            return 0.0

        upper = centerline + self._multiplier * atr
        current = float(price_history[-1])

        if current <= upper:
            return 0.0

        excess_atr = (current - upper) / atr
        return self.clamp(excess_atr, 0.0, 1.0)

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return (
            f"Bullish when price breaks above the {self._window}-bar SMA + "
            f"{self._multiplier:.1f} x ATR upper Keltner band"
        )
