"""VWAP reversion strategy: buy when the close trades below rolling VWAP.

Classical VWAP weights each bar's typical price ``(H + L + C) / 3`` by
its traded volume. When the provider has no volume data (or the window
traded zero contracts), this strategy degenerates to the simple moving
average of the typical price — equivalent to the prior close-SMA proxy
for the close-only fixtures used in tests.
"""

from __future__ import annotations

import numpy as np

from midas.data.price_history import PriceHistory
from midas.models import AssetSuitability
from midas.strategies.base import EntrySignal


class VWAPReversion(EntrySignal):
    def __init__(self, window: int = 20, threshold: float = 0.02) -> None:
        self._window = window
        self._threshold = threshold

    @property
    def warmup_period(self) -> int:
        return self._window

    def _rolling_vwap(self, price_history: PriceHistory) -> np.ndarray | None:
        """Rolling VWAP of length ``len(close) - window + 1``, or None.

        Returns None only when the history is shorter than the window.
        When volume data is missing — or the window's total volume is
        zero — the bars in question fall back to the simple mean of the
        typical price.
        """
        close = price_history.close
        n = len(close)
        w = self._window
        if n < w:
            return None
        typical = (price_history.high + price_history.low + close) / 3.0
        cs_t = np.empty(n + 1)
        cs_t[0] = 0.0
        np.cumsum(typical, out=cs_t[1:])
        typical_sum = cs_t[w:] - cs_t[:-w]
        if price_history.volume is None:
            return typical_sum / w
        volume = price_history.volume
        cs_pv = np.empty(n + 1)
        cs_pv[0] = 0.0
        np.cumsum(typical * volume, out=cs_pv[1:])
        cs_v = np.empty(n + 1)
        cs_v[0] = 0.0
        np.cumsum(volume, out=cs_v[1:])
        pv_sum = cs_pv[w:] - cs_pv[:-w]
        v_sum = cs_v[w:] - cs_v[:-w]
        sma = typical_sum / w
        return np.where(v_sum > 0, pv_sum / np.where(v_sum > 0, v_sum, 1.0), sma)

    def precompute(self, price_history: PriceHistory) -> np.ndarray | None:
        close = price_history.close
        n = len(close)
        w = self._window
        scores = np.full(n, np.nan)
        if n < w:
            return scores
        vwap = self._rolling_vwap(price_history)
        assert vwap is not None  # n >= w
        current = close[w - 1 :]
        with np.errstate(divide="ignore", invalid="ignore"):
            deviation = np.where(vwap != 0, (current - vwap) / vwap, 0.0)
        scores[w - 1 :] = np.clip(-deviation / self._threshold, 0.0, 1.0)
        return scores

    def score(
        self,
        price_history: PriceHistory,
        **kwargs: object,
    ) -> float | None:
        if len(price_history.close) < self._window:
            return None
        vwap = self._rolling_vwap(price_history)
        assert vwap is not None
        vwap_now = float(vwap[-1])
        if vwap_now == 0:
            return 0.0
        current = float(price_history.close[-1])
        deviation = (current - vwap_now) / vwap_now
        # Buy-only entry signal: negative deviation (below VWAP) ramps from 0
        # to 1. The bearish "above VWAP" half is dropped — exits are handled
        # by ExitRule strategies.
        return self.clamp(-deviation / self._threshold, 0.0, 1.0)

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.LARGE_CAP, AssetSuitability.BROAD_MARKET_ETF]

    @property
    def description(self) -> str:
        return f"Bullish when price trades {self._threshold:.0%} below the {self._window}-bar rolling VWAP"
