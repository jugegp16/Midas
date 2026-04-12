"""Parabolic SAR exit: Wilder's self-accelerating trailing stop."""

from __future__ import annotations

import numpy as np

from midas.data.price_history import PriceHistory
from midas.models import AssetSuitability
from midas.strategies.base import ExitRule


class ParabolicSARExit(ExitRule):
    def __init__(
        self,
        af_start: float = 0.02,
        af_step: float = 0.02,
        af_max: float = 0.20,
    ) -> None:
        self._af_start = af_start
        self._af_step = af_step
        self._af_max = af_max

    @property
    def warmup_period(self) -> int:
        return 2

    def _compute(
        self,
        high: np.ndarray,
        low: np.ndarray,
    ) -> tuple[float, bool] | None:
        """Return (sar, uptrend) at the final bar, or None if too little data.

        Wilder's spec uses HIGH for the uptrend extreme point and LOW for the
        downtrend extreme point, and flips the trend when SAR crosses LOW
        (uptrend) or HIGH (downtrend).
        """
        num_bars = len(high)
        if num_bars < 2:
            return None

        h0, h1 = float(high[0]), float(high[1])
        l0, l1 = float(low[0]), float(low[1])
        uptrend = h1 >= h0
        if uptrend:
            sar = min(l0, l1)
            ep = max(h0, h1)
        else:
            sar = max(h0, h1)
            ep = min(l0, l1)
        af = self._af_start

        for i in range(2, num_bars):
            sar = sar + af * (ep - sar)
            hi = float(high[i])
            lo = float(low[i])
            if uptrend:
                if hi > ep:
                    ep = hi
                    af = min(af + self._af_step, self._af_max)
                if sar > lo:
                    uptrend = False
                    sar = ep
                    ep = lo
                    af = self._af_start
            else:
                if lo < ep:
                    ep = lo
                    af = min(af + self._af_step, self._af_max)
                if sar < hi:
                    uptrend = True
                    sar = ep
                    ep = hi
                    af = self._af_start

        return sar, uptrend

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
        result = self._compute(price_history.high, price_history.low)
        if result is None:
            return proposed_target
        _sar, uptrend = result
        if not uptrend:
            return 0.0
        return proposed_target

    def clamp_reason(
        self,
        ticker: str,
        price_history: PriceHistory,
        cost_basis: float,
        high_water_mark: float,
    ) -> str:
        close = price_history.close
        result = self._compute(price_history.high, price_history.low)
        current = float(close[-1]) if len(close) > 0 else 0.0
        if result is None:
            return f"ParabolicSARExit: flip on {ticker}"
        sar, _ = result
        return f"ParabolicSARExit: SAR ${sar:.2f} flipped above price ${current:.2f} (downtrend)"

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return f"Sell when Wilder's Parabolic SAR (AF {self._af_start:.2f}->{self._af_max:.2f}) flips above price"
