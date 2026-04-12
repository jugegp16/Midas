"""Gap down recovery strategy: buy when price gaps down then recovers."""

from __future__ import annotations

import numpy as np

from midas.data.price_history import PriceHistory
from midas.models import AssetSuitability
from midas.strategies.base import EntrySignal


class GapDownRecovery(EntrySignal):
    def __init__(self, gap_threshold: float = 0.03) -> None:
        self._gap_threshold = gap_threshold

    @property
    def warmup_period(self) -> int:
        return 2

    def precompute(self, price_history: PriceHistory) -> np.ndarray | None:
        close = price_history.close
        opens = price_history.open
        num_bars = len(close)
        scores = np.full(num_bars, np.nan)
        if num_bars < 2:
            return scores
        prev_close = close[:-1]
        today_open = opens[1:]
        current = close[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            gap_pct = np.where(prev_close != 0, (prev_close - today_open) / prev_close, 0.0)
            is_gap = gap_pct >= self._gap_threshold
            is_recovery = current > today_open
            active = is_gap & is_recovery
            gap_size = prev_close - today_open
            recovery_pct = np.where(
                active & (gap_size != 0),
                (current - today_open) / gap_size,
                0.0,
            )
        scores[1:] = np.where(active, np.clip(np.minimum(recovery_pct, 1.0), 0.0, 1.0), 0.0)
        return scores

    def score(
        self,
        price_history: PriceHistory,
        **kwargs: object,
    ) -> float | None:
        close = price_history.close
        if len(close) < 2:
            return None

        prev_close = float(close[-2])
        today_open = float(price_history.open[-1])
        current = float(close[-1])

        if prev_close == 0:
            return 0.0

        gap_pct = (prev_close - today_open) / prev_close

        if gap_pct >= self._gap_threshold and current > today_open:
            recovery_pct = (current - today_open) / (prev_close - today_open)
            return self.clamp(min(recovery_pct, 1.0), 0.0, 1.0)
        return 0.0

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.INDIVIDUAL_EQUITY, AssetSuitability.HIGH_VOLATILITY]

    @property
    def description(self) -> str:
        return f"Buy when price gaps down >{self._gap_threshold:.0%} then recovers above the gap level"
