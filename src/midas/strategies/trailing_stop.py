"""Trailing stop strategy: sell when price falls from high-water mark."""

from __future__ import annotations

import numpy as np
import pandas as pd

from midas.models import AssetSuitability, StrategyTier
from midas.strategies.base import Strategy


class TrailingStop(Strategy):
    def __init__(self, trail_pct: float = 0.10) -> None:
        self._trail_pct = trail_pct

    @property
    def tier(self) -> StrategyTier:
        return StrategyTier.PROTECTIVE

    def score(
        self,
        price_history: pd.Series,
        *,
        cost_basis: float | None = None,
        **kwargs: object,
    ) -> float | None:
        if cost_basis is None or cost_basis <= 0:
            return None

        if len(price_history) < 2:
            return None

        values = np.asarray(price_history)
        current = float(values[-1])
        high_water = float(max(values.max(), cost_basis))

        if high_water == 0:
            return 0.0

        drawdown = (high_water - current) / high_water

        if drawdown >= self._trail_pct and current > cost_basis:
            return -self._clamp(
                (drawdown - self._trail_pct) / self._trail_pct
            )
        return 0.0

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return (
            f"Sell when price falls {self._trail_pct:.0%} from its "
            f"high-water mark (while still above cost basis)"
        )
