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

    def _compute(self, prices: np.ndarray) -> tuple[float, bool] | None:
        """Return (sar, uptrend) at the final bar, or None if too little data."""
        n = len(prices)
        if n < 2:
            return None

        p0 = float(prices[0])
        p1 = float(prices[1])
        uptrend = p1 >= p0
        if uptrend:
            sar = min(p0, p1)
            ep = max(p0, p1)
        else:
            sar = max(p0, p1)
            ep = min(p0, p1)
        af = self._af_start

        for i in range(2, n):
            sar = sar + af * (ep - sar)
            close = float(prices[i])
            if uptrend:
                if close > ep:
                    ep = close
                    af = min(af + self._af_step, self._af_max)
                if sar > close:
                    uptrend = False
                    sar = ep
                    ep = close
                    af = self._af_start
            else:
                if close < ep:
                    ep = close
                    af = min(af + self._af_step, self._af_max)
                if sar < close:
                    uptrend = True
                    sar = ep
                    ep = close
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
        result = self._compute(price_history.close)
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
        prices = price_history.close
        result = self._compute(prices)
        current = float(prices[-1]) if len(prices) > 0 else 0.0
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
