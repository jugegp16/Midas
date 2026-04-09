"""Profit taking exit rule: sell lots whose unrealized gain exceeds threshold."""

from __future__ import annotations

import numpy as np

from midas.models import AssetSuitability, ExitIntent, PositionLot
from midas.strategies.base import ExitRule


class ProfitTaking(ExitRule):
    def __init__(self, gain_threshold: float = 0.20) -> None:
        self._gain_threshold = gain_threshold

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

        def triggered(lot: PositionLot) -> bool:
            return (current - lot.cost_basis) / lot.cost_basis >= self._gain_threshold

        def reason(lot: PositionLot) -> str:
            gain_pct = (current - lot.cost_basis) / lot.cost_basis
            return (
                f"{lot.shares:g} shares at {gain_pct:.1%} gain "
                f"vs cost basis ${lot.cost_basis:.2f} (threshold {self._gain_threshold:.0%})"
            )

        return self.fire_on_lots(ticker, lots, current, triggered, reason)

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return f"Sell lots whose unrealized gain exceeds {self._gain_threshold:.0%} of cost basis"
