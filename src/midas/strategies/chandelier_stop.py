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

    def _wilder_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> float:
        """Wilder-smoothed ATR over the full available history.

        TR_t = max(H-L, |H - prevC|, |L - prevC|), with bar 0 falling back
        to H - L since no prior close is available. The seed ATR is the
        simple mean of the first ``window`` TRs; subsequent bars apply the
        recursive formula ``ATR_t = ((window - 1) * ATR_{t-1} + TR_t) / window``.

        With exactly ``window`` bars of history this degenerates to the
        simple mean (no iteration). With more, the exponential decay tail
        matches Wilder's original definition and most charting tools.
        """
        num_bars = len(close)
        tr = np.empty(num_bars)
        tr[0] = high[0] - low[0]
        if num_bars > 1:
            prev_c = close[:-1]
            tr[1:] = np.maximum.reduce([high[1:] - low[1:], np.abs(high[1:] - prev_c), np.abs(low[1:] - prev_c)])
        window = self._window
        atr = float(tr[:window].mean())
        for idx in range(window, num_bars):
            atr = ((window - 1) * atr + float(tr[idx])) / window
        return atr

    def _stop_level(self, price_history: PriceHistory) -> tuple[float, float, float] | None:
        """Return ``(stop, highest_high, atr)`` or None if not enough history."""
        close = price_history.close
        if len(close) < self._window:
            return None
        atr = self._wilder_atr(price_history.high, price_history.low, close)
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
