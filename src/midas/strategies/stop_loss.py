"""Stop loss exit rule: sell lots whose unrealized loss exceeds threshold."""

from __future__ import annotations

import numpy as np

from midas.models import AssetSuitability, ExitIntent, PositionLot
from midas.strategies.base import ExitRule


class StopLoss(ExitRule):
    def __init__(self, loss_threshold: float = 0.10) -> None:
        self._loss_threshold = loss_threshold

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
            return (lot.cost_basis - current) / lot.cost_basis >= self._loss_threshold

        def reason(lot: PositionLot) -> str:
            loss_pct = (lot.cost_basis - current) / lot.cost_basis
            return (
                f"{lot.shares:g} shares at {loss_pct:.1%} loss "
                f"vs cost basis ${lot.cost_basis:.2f} (threshold {self._loss_threshold:.0%})"
            )

        return self.fire_on_lots(ticker, lots, current, triggered, reason)

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return f"Sell lots whose unrealized loss exceeds {self._loss_threshold:.0%} of cost basis"
