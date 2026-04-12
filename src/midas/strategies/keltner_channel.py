"""Keltner Channel breakout: bullish when price breaks above SMA + k x ATR."""

from __future__ import annotations

import numpy as np

from midas.data.price_history import PriceHistory
from midas.models import AssetSuitability
from midas.strategies.base import EntrySignal


def _true_range_series(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Full per-bar TR series of length ``len(close)``.

    Bar 0 has no prior close available, so its TR falls back to H - L
    (Wilder convention). Bars 1..n-1 use max(H-L, |H-prevC|, |L-prevC|).
    """
    num_bars = len(close)
    tr = np.empty(num_bars)
    tr[0] = high[0] - low[0]
    if num_bars > 1:
        prev_c = close[:-1]
        tr[1:] = np.maximum.reduce([high[1:] - low[1:], np.abs(high[1:] - prev_c), np.abs(low[1:] - prev_c)])
    return tr


class KeltnerChannel(EntrySignal):
    def __init__(self, window: int = 20, multiplier: float = 2.0) -> None:
        self._window = window
        self._multiplier = multiplier

    @property
    def warmup_period(self) -> int:
        return self._window

    def precompute(self, price_history: PriceHistory) -> np.ndarray | None:
        close = price_history.close
        num_bars = len(close)
        window = self._window
        scores = np.full(num_bars, np.nan)
        if num_bars < window or window < 2:
            return scores
        cs = np.empty(num_bars + 1)
        cs[0] = 0.0
        np.cumsum(close, out=cs[1:])
        centerlines = (cs[window:] - cs[:-window]) / window
        tr = _true_range_series(price_history.high, price_history.low, close)
        cs_tr = np.empty(num_bars + 1)
        cs_tr[0] = 0.0
        np.cumsum(tr, out=cs_tr[1:])
        atr_series = (cs_tr[window:] - cs_tr[:-window]) / window
        upper = centerlines + self._multiplier * atr_series
        current = close[window - 1 :]
        with np.errstate(divide="ignore", invalid="ignore"):
            excess_atr = np.where(atr_series > 0, (current - upper) / atr_series, 0.0)
        raw = np.where(current > upper, excess_atr, 0.0)
        scores[window - 1 :] = np.clip(raw, 0.0, 1.0)
        return scores

    def score(
        self,
        price_history: PriceHistory,
        **kwargs: object,
    ) -> float | None:
        close = price_history.close
        window = self._window
        if len(close) < window:
            return None

        start = len(close) - window
        recent = close[start:]
        centerline = float(recent.mean())

        high = price_history.high
        low = price_history.low
        high_slice = high[start:]
        low_slice = low[start:]
        if start >= 1:
            prev_c = close[start - 1 : -1]
            tr = np.maximum.reduce([high_slice - low_slice, np.abs(high_slice - prev_c), np.abs(low_slice - prev_c)])
        else:
            tr = np.empty(window)
            tr[0] = high_slice[0] - low_slice[0]
            prev_c = close[:-1]
            tr[1:] = np.maximum.reduce(
                [
                    high_slice[1:] - low_slice[1:],
                    np.abs(high_slice[1:] - prev_c),
                    np.abs(low_slice[1:] - prev_c),
                ]
            )
        atr = float(tr.mean())

        if atr <= 0:
            return 0.0

        upper = centerline + self._multiplier * atr
        current = float(close[-1])

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
