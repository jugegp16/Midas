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
