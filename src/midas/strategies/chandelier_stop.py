"""Chandelier exit: k x ATR trailing stop off the rolling N-bar highest high."""

from __future__ import annotations

import numpy as np

from midas.data.price_history import PriceHistory
from midas.models import AssetSuitability
from midas.strategies.base import ExitRule


class ChandelierStop(ExitRule):
    def __init__(self, window: int = 22, multiplier: float = 3.0) -> None:
        self._window = window
        self._multiplier = multiplier

    @property
    def warmup_period(self) -> int:
        return self._window

    def _true_range(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> np.ndarray:
        """TR over the last N bars: max(H-L, |H-prevC|, |L-prevC|).

        When the window's first bar has no prior close available, its TR
        falls back to H - L (standard Wilder convention).
        """
        n = self._window
        start = len(close) - n
        h = high[start:]
        lo = low[start:]
        if start >= 1:
            prev_c = close[start - 1 : -1]
            tr = np.maximum.reduce([h - lo, np.abs(h - prev_c), np.abs(lo - prev_c)])
        else:
            tr = np.empty(n)
            tr[0] = h[0] - lo[0]
            prev_c = close[:-1]
            tr[1:] = np.maximum.reduce([h[1:] - lo[1:], np.abs(h[1:] - prev_c), np.abs(lo[1:] - prev_c)])
        return np.asarray(tr, dtype=float)

    def _stop_level(self, price_history: PriceHistory) -> tuple[float, float, float] | None:
        """Return ``(stop, highest_high, atr)`` or None if not enough history."""
        close = price_history.close
        if len(close) < self._window:
            return None
        tr = self._true_range(price_history.high, price_history.low, close)
        atr = float(tr.mean())
        highest = float(price_history.high[-self._window :].max())
        return highest - self._multiplier * atr, highest, atr

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
        level = self._stop_level(price_history)
        if level is None:
            return proposed_target
        stop, _, _ = level
        current = float(price_history.close[-1])
        if current < stop:
            return 0.0
        return proposed_target

    def clamp_reason(
        self,
        ticker: str,
        price_history: PriceHistory,
        cost_basis: float,
        high_water_mark: float,
    ) -> str:
        level = self._stop_level(price_history)
        current = float(price_history.close[-1])
        if level is None:
            return f"ChandelierStop: insufficient history for {ticker}"
        stop, highest, atr = level
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
