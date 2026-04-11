"""Chandelier exit: k x ATR trailing stop off the rolling N-bar highest close."""

from __future__ import annotations

import numpy as np

from midas.models import AssetSuitability
from midas.strategies.base import ExitRule


class ChandelierStop(ExitRule):
    def __init__(self, window: int = 22, multiplier: float = 3.0) -> None:
        self._window = window
        self._multiplier = multiplier

    @property
    def warmup_period(self) -> int:
        return self._window

    def _stop_level(self, price_history: np.ndarray) -> float | None:
        if len(price_history) < self._window:
            return None
        recent = price_history[-self._window :]
        highest = float(recent.max())
        atr = float(np.abs(np.diff(recent)).mean())
        return highest - self._multiplier * atr

    def clamp_target(
        self,
        ticker: str,
        proposed_target: float,
        price_history: np.ndarray,
        cost_basis: float,
        high_water_mark: float,
    ) -> float:
        if proposed_target <= 0:
            return proposed_target
        stop = self._stop_level(price_history)
        if stop is None:
            return proposed_target
        current = float(price_history[-1])
        if current < stop:
            return 0.0
        return proposed_target

    def clamp_reason(
        self,
        ticker: str,
        price_history: np.ndarray,
        cost_basis: float,
        high_water_mark: float,
    ) -> str:
        recent = price_history[-self._window :]
        highest = float(recent.max())
        atr = float(np.abs(np.diff(recent)).mean()) if len(recent) > 1 else 0.0
        stop = highest - self._multiplier * atr
        current = float(price_history[-1])
        return (
            f"ChandelierStop: price ${current:.2f} below "
            f"${stop:.2f} (${highest:.2f} peak - "
            f"{self._multiplier:.1f} x ${atr:.2f} ATR{self._window})"
        )

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return (
            f"Sell when price falls {self._multiplier:.1f} x ATR below the "
            f"rolling {self._window}-bar high (volatility-adjusted trailing stop)"
        )
