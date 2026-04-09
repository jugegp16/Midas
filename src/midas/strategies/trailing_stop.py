"""Trailing stop exit rule: sell when price falls from per-lot high-water mark."""

from __future__ import annotations

import numpy as np

from midas.models import AssetSuitability, ExitIntent, PositionLot
from midas.strategies.base import ExitRule


class TrailingStop(ExitRule):
    def __init__(self, trail_pct: float = 0.10) -> None:
        self._trail_pct = trail_pct

    def evaluate_exit(
        self,
        ticker: str,
        lots: list[PositionLot],
        price_history: np.ndarray,
    ) -> list[ExitIntent]:
        if not lots or len(price_history) == 0:
            return []
        current = float(price_history[-1])
        if current <= 0:
            return []

        # Each lot tracks its own high-water mark since purchase, updated by
        # the execution engine on every tick. A lot only triggers if (a) the
        # trailing drawdown threshold is met AND (b) the lot is currently in
        # profit — the trailing stop is a gain-protection mechanism, not a
        # loss-cut (StopLoss handles that). Lots without a recorded high
        # (synthesized lots in live mode before any tick) fall back to cost
        # basis, which keeps the rule from firing until the engine has had a
        # chance to observe at least one price.
        def triggered(lot: PositionLot) -> bool:
            high_water = lot.high_water_mark if lot.high_water_mark is not None else lot.cost_basis
            if high_water <= 0:
                return False
            drawdown = (high_water - current) / high_water
            return drawdown >= self._trail_pct and current > lot.cost_basis

        def reason(lot: PositionLot) -> str:
            high_water = lot.high_water_mark if lot.high_water_mark is not None else lot.cost_basis
            drawdown_pct = (high_water - current) / high_water if high_water else 0.0
            return (
                f"{lot.shares:g} shares: drawdown {drawdown_pct:.1%} from high "
                f"${high_water:.2f} (threshold {self._trail_pct:.0%}, basis ${lot.cost_basis:.2f})"
            )

        return self.fire_on_lots(ticker, lots, current, triggered, reason)

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return f"Sell lots in profit when price falls {self._trail_pct:.0%} from their high-water mark"
