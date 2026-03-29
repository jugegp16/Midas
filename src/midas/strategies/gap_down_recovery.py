"""Gap down recovery strategy: buy when price gaps down then recovers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from midas.models import AssetSuitability
from midas.strategies.base import Strategy


class GapDownRecovery(Strategy):
    def __init__(self, gap_threshold: float = 0.03) -> None:
        self._gap_threshold = gap_threshold

    def score(
        self,
        ticker: str,
        price_history: pd.Series,
        **kwargs: object,
    ) -> float | None:
        if len(price_history) < 3:
            return None

        values = np.asarray(price_history)
        prev_close = float(values[-3])
        gap_open = float(values[-2])
        current = float(values[-1])

        if prev_close == 0:
            return 0.0

        gap_pct = (prev_close - gap_open) / prev_close

        if gap_pct >= self._gap_threshold and current > gap_open:
            recovery_pct = (current - gap_open) / (prev_close - gap_open)
            return self._clamp(min(recovery_pct, 1.0))
        return 0.0

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.INDIVIDUAL_EQUITY, AssetSuitability.HIGH_VOLATILITY]

    @property
    def description(self) -> str:
        return (
            f"Buy when price gaps down >{self._gap_threshold:.0%} "
            f"then recovers above the gap level"
        )
