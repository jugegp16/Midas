"""Trailing stop strategy: sell when price falls from high-water mark."""

from __future__ import annotations

import numpy as np
import pandas as pd

from midas.models import AssetSuitability, Direction, Signal
from midas.strategies.base import Strategy


class TrailingStop(Strategy):
    def __init__(self, trail_pct: float = 0.10) -> None:
        self._trail_pct = trail_pct

    def evaluate(
        self,
        ticker: str,
        price_history: pd.Series,
        *,
        cost_basis: float | None = None,
        **kwargs: object,
    ) -> list[Signal]:
        if cost_basis is None or cost_basis <= 0:
            return []

        if len(price_history) < 2:
            return []

        values = np.asarray(price_history)
        current = float(values[-1])

        # High-water mark since entry: max of price history and cost basis
        high_water = float(max(values.max(), cost_basis))

        if high_water == 0:
            return []

        drawdown = (high_water - current) / high_water

        if drawdown >= self._trail_pct and current > cost_basis:
            return [self._make_signal(
                ticker,
                Direction.SELL,
                strength=(drawdown - self._trail_pct) / self._trail_pct,
                reasoning=(
                    f"{ticker} down {drawdown:.1%} from high of ${high_water:.2f}, "
                    f"trailing stop {self._trail_pct:.0%} triggered at ${current:.2f}"
                ),
                price=current,
            )]
        return []

    @property
    def name(self) -> str:
        return f"TrailingStop(trail_pct={self._trail_pct})"

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return (
            f"Sell when price falls {self._trail_pct:.0%} from its "
            f"high-water mark (while still above cost basis)"
        )
