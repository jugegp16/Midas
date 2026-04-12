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
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    if n > 1:
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
        n = len(close)
        w = self._window
        scores = np.full(n, np.nan)
        if n < w or w < 2:
            return scores
        cs = np.empty(n + 1)
        cs[0] = 0.0
        np.cumsum(close, out=cs[1:])
        centerlines = (cs[w:] - cs[:-w]) / w
        tr = _true_range_series(price_history.high, price_history.low, close)
        cs_tr = np.empty(n + 1)
        cs_tr[0] = 0.0
        np.cumsum(tr, out=cs_tr[1:])
        atr_series = (cs_tr[w:] - cs_tr[:-w]) / w
        upper = centerlines + self._multiplier * atr_series
        current = close[w - 1 :]
        with np.errstate(divide="ignore", invalid="ignore"):
            excess_atr = np.where(atr_series > 0, (current - upper) / atr_series, 0.0)
        raw = np.where(current > upper, excess_atr, 0.0)
        scores[w - 1 :] = np.clip(raw, 0.0, 1.0)
        return scores

    def score(
        self,
        price_history: PriceHistory,
        **kwargs: object,
    ) -> float | None:
        close = price_history.close
        n = self._window
        if len(close) < n:
            return None

        start = len(close) - n
        recent = close[start:]
        centerline = float(recent.mean())

        high = price_history.high
        low = price_history.low
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
