"""Profit taking strategy: sell when unrealized gains exceed threshold."""

from __future__ import annotations

import numpy as np

from midas.models import AssetSuitability
from midas.strategies.base import Strategy


class ProfitTaking(Strategy):
    def __init__(self, gain_threshold: float = 0.20) -> None:
        self._gain_threshold = gain_threshold

    def score(
        self,
        price_history: np.ndarray,
        *,
        cost_basis: float | None = None,
        **kwargs: object,
    ) -> float | None:
        if cost_basis is None or cost_basis <= 0:
            return None

        current = float(price_history[-1])
        gain = (current - cost_basis) / cost_basis

        if gain >= self._gain_threshold:
            return -self._clamp((gain - self._gain_threshold) / self._gain_threshold)
        return 0.0

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return f"Sell when unrealized gains exceed {self._gain_threshold:.0%} of cost basis"
