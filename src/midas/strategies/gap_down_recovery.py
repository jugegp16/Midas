"""Gap down recovery strategy: buy when price gaps down then recovers."""

from __future__ import annotations

import numpy as np

from midas.models import AssetSuitability
from midas.strategies.base import EntrySignal


class GapDownRecovery(EntrySignal):
    def __init__(self, gap_threshold: float = 0.03) -> None:
        self._gap_threshold = gap_threshold

    @property
    def warmup_period(self) -> int:
        return 3

    def precompute(self, prices: np.ndarray) -> np.ndarray | None:
        n = len(prices)
        scores = np.full(n, np.nan)
        if n < 3:
            return scores
        prev_close = prices[:-2]
        gap_open = prices[1:-1]
        current = prices[2:]
        gap_pct = np.where(prev_close != 0, (prev_close - gap_open) / prev_close, 0.0)
        is_gap = gap_pct >= self._gap_threshold
        is_recovery = current > gap_open
        active = is_gap & is_recovery
        gap_size = prev_close - gap_open
        recovery_pct = np.where(
            active & (gap_size != 0),
            (current - gap_open) / gap_size,
            0.0,
        )
        scores[2:] = np.where(active, np.clip(np.minimum(recovery_pct, 1.0), 0.0, 1.0), 0.0)
        return scores

    def score(
        self,
        price_history: np.ndarray,
        **kwargs: object,
    ) -> float | None:
        if len(price_history) < 3:
            return None

        prev_close = float(price_history[-3])
        gap_open = float(price_history[-2])
        current = float(price_history[-1])

        if prev_close == 0:
            return 0.0

        gap_pct = (prev_close - gap_open) / prev_close

        if gap_pct >= self._gap_threshold and current > gap_open:
            recovery_pct = (current - gap_open) / (prev_close - gap_open)
            return self.clamp(min(recovery_pct, 1.0), 0.0, 1.0)
        return 0.0

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.INDIVIDUAL_EQUITY, AssetSuitability.HIGH_VOLATILITY]

    @property
    def description(self) -> str:
        return f"Buy when price gaps down >{self._gap_threshold:.0%} then recovers above the gap level"
