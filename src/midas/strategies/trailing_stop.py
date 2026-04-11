"""Trailing stop exit rule: sell when price falls from high-water mark."""

from __future__ import annotations

import numpy as np

from midas.models import AssetSuitability
from midas.strategies.base import ExitRule


class TrailingStop(ExitRule):
    def __init__(self, trail_pct: float = 0.10) -> None:
        self._trail_pct = trail_pct

    def clamp_target(
        self,
        ticker: str,
        proposed_target: float,
        price_history: np.ndarray,
        cost_basis: float,
        high_water_mark: float,
    ) -> float:
        if proposed_target <= 0 or len(price_history) == 0:
            return proposed_target
        current = float(price_history[-1])
        if current <= 0 or high_water_mark <= 0:
            return proposed_target
        drawdown = (high_water_mark - current) / high_water_mark
        in_profit = current > cost_basis if cost_basis > 0 else False
        if drawdown >= self._trail_pct and in_profit:
            return 0.0
        return proposed_target

    def clamp_reason(
        self,
        ticker: str,
        price_history: np.ndarray,
        cost_basis: float,
        high_water_mark: float,
    ) -> str:
        current = float(price_history[-1])
        drawdown = (high_water_mark - current) / high_water_mark if high_water_mark > 0 else 0.0
        return (
            f"TrailingStop: {drawdown:.1%} drawdown from high "
            f"${high_water_mark:.2f} (threshold {self._trail_pct:.0%}, "
            f"basis ${cost_basis:.2f})"
        )

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return f"Sell when price falls {self._trail_pct:.0%} from high-water mark (gain-protection only)"
