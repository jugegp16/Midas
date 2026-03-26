"""Bollinger Band strategy: buy when price touches lower band."""

from __future__ import annotations

import numpy as np
import pandas as pd

from midas.models import AssetSuitability, Direction, Signal
from midas.strategies.base import Strategy


class BollingerBand(Strategy):
    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        self._window = window
        self._num_std = num_std

    def evaluate(
        self,
        ticker: str,
        price_history: pd.Series,
        **kwargs: object,
    ) -> list[Signal]:
        if len(price_history) < self._window:
            return []

        values = np.asarray(price_history)
        window_data = values[-self._window :]
        ma = float(window_data.mean())
        std = float(window_data.std(ddof=1))

        if std == 0:
            return []

        lower_band = ma - self._num_std * std
        current = float(values[-1])

        if current <= lower_band:
            pct_below = (lower_band - current) / std
            return [self._make_signal(
                ticker,
                Direction.BUY,
                strength=pct_below / self._num_std,
                reasoning=(
                    f"{ticker} at ${current:.2f} touched lower Bollinger Band "
                    f"(${lower_band:.2f}), {self._window}-day MA ${ma:.2f} "
                    f"± {self._num_std}σ"
                ),
                price=current,
            )]
        return []

    @property
    def name(self) -> str:
        return f"BollingerBand(window={self._window}, num_std={self._num_std})"

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.BROAD_MARKET_ETF, AssetSuitability.LARGE_CAP]

    @property
    def description(self) -> str:
        return (
            f"Buy when price touches the lower Bollinger Band "
            f"({self._window}-day MA − {self._num_std}σ)"
        )
