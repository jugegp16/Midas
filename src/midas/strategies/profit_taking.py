"""Profit taking exit rule: sell when unrealized gain exceeds threshold."""

from __future__ import annotations

from midas.data.price_history import PriceHistory
from midas.models import AssetSuitability
from midas.strategies.base import ExitRule


class ProfitTaking(ExitRule):
    def __init__(self, gain_threshold: float = 0.20) -> None:
        self._gain_threshold = gain_threshold

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
        gain = (current - cost_basis) / cost_basis
        if gain >= self._gain_threshold:
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
        gain = (current - cost_basis) / cost_basis
        return f"ProfitTaking: {gain:.1%} gain vs cost basis ${cost_basis:.2f} (threshold {self._gain_threshold:.0%})"

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return f"Sell when unrealized gain exceeds {self._gain_threshold:.0%} of cost basis"
