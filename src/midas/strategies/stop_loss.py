"""Stop loss exit rule: sell when unrealized loss exceeds threshold."""

from __future__ import annotations

from midas.data.price_history import PriceHistory
from midas.models import AssetSuitability
from midas.strategies.base import ExitRule


class StopLoss(ExitRule):
    def __init__(self, loss_threshold: float = 0.10) -> None:
        self._loss_threshold = loss_threshold

    def clamp_target(
        self,
        ticker: str,
        proposed_target: float,
        price_history: PriceHistory,
        cost_basis: float,
        high_water_mark: float,
    ) -> float:
        if proposed_target <= 0 or len(price_history) == 0 or cost_basis <= 0:
            return proposed_target
        current = float(price_history.close[-1])
        if current <= 0:
            return proposed_target
        loss = (cost_basis - current) / cost_basis
        if loss >= self._loss_threshold:
            return 0.0
        return proposed_target

    def clamp_reason(
        self,
        ticker: str,
        price_history: PriceHistory,
        cost_basis: float,
        high_water_mark: float,
    ) -> str:
        current = float(price_history.close[-1])
        loss = (cost_basis - current) / cost_basis
        return f"StopLoss: {loss:.1%} loss vs cost basis ${cost_basis:.2f} (threshold {self._loss_threshold:.0%})"

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return f"Sell when unrealized loss exceeds {self._loss_threshold:.0%} of cost basis"
