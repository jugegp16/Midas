"""Abstract strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np

from midas.models import AssetSuitability, MechanicalIntent, StrategyTier

# Recursive indicators (EMA, RSI, MACD) converge to their steady-state value
# only after ~3-10x their nominal period. Following TA-Lib's "Unstable Period"
# convention, we use 4x as the conservative default so strategies don't score
# on numerically unstable values during warmup.
RECURSIVE_WARMUP_MULTIPLIER = 4

# Convert warmup bars (trading days) to calendar days. ~252 trading days per
# year vs 365 calendar days gives 1.45x; rounded up to 1.5x for safety.
TRADING_TO_CALENDAR_RATIO = 1.5
# Extra calendar days added on top of the conversion to cover holidays and
# weekends that would otherwise clip the warmup window.
WARMUP_CALENDAR_SLACK = 10
# Minimum calendar-day buffer to always fetch, so strategies with tiny windows
# still get a sensible prefix and we don't re-derive a near-zero buffer.
MIN_WARMUP_CALENDAR_DAYS = 30


def max_warmup(strategies: Iterable[Strategy]) -> int:
    """Largest ``warmup_period`` across an iterable of strategies (0 if empty)."""
    return max((s.warmup_period for s in strategies), default=0)


def warmup_bars_to_calendar_days(bars: int) -> int:
    """Convert a trading-day warmup requirement to a calendar-day buffer.

    Always returns at least ``MIN_WARMUP_CALENDAR_DAYS`` so callers (notably
    ``LiveEngine``) get a sensible fetch window even when no configured
    strategy advertises a warmup requirement.
    """
    bars = max(bars, 0)
    calendar = int(bars * TRADING_TO_CALENDAR_RATIO) + WARMUP_CALENDAR_SLACK
    return max(calendar, MIN_WARMUP_CALENDAR_DAYS)


class Strategy(ABC):
    """Base class for all signal strategies.

    Strategies are stateless and ticker-agnostic. They receive price history
    and return a conviction score.
    """

    @abstractmethod
    def score(
        self,
        price_history: np.ndarray,
        *,
        cost_basis: float | None = None,
        **kwargs: object,
    ) -> float | None:
        """Return conviction score.

        Positive = bullish, negative = bearish, 0 = neutral, None = abstain.
        Typically in [-1, +1].
        """

    @property
    def tier(self) -> StrategyTier:
        """Strategy tier — override in subclass if not CONVICTION."""
        return StrategyTier.CONVICTION

    @property
    def warmup_period(self) -> int:
        """Bars of price history required before this strategy produces valid scores.

        Backtest/optimize/live use ``max(warmup_period)`` across configured
        strategies to prefetch a lookback buffer before the user's start date,
        so strategies can emit valid signals from day one of the simulation
        rather than spending the first N days in warmup.

        Default is 0 (no warmup — e.g. mechanical or position-only strategies).
        Override in subclasses that depend on a rolling window. For recursive
        indicators (EMA/RSI/MACD) multiply the nominal period by
        ``RECURSIVE_WARMUP_MULTIPLIER`` so the indicator has room to converge
        to its steady-state value.
        """
        return 0

    @staticmethod
    def clamp(value: float, lo: float, hi: float) -> float:
        """Clamp *value* into [lo, hi]."""
        return max(lo, min(hi, value))

    def precompute(self, prices: np.ndarray) -> np.ndarray | None:
        """Precompute scores for every prefix of *prices* in one pass.

        Returns an array *s* of length ``len(prices)`` where ``s[i]`` equals
        ``self.score(prices[:i+1])`` (or ``NaN`` when ``score`` would return
        ``None``).  Returns ``None`` when precomputation is not possible
        (e.g. the strategy needs runtime context like ``cost_basis``).
        """
        return None

    def generate_intents(
        self,
        ticker: str,
        price_history: np.ndarray,
        **kwargs: object,
    ) -> list[MechanicalIntent]:
        """Return mechanical order intents. Override for MECHANICAL strategies."""
        return []

    @property
    def name(self) -> str:
        """Human-readable strategy name."""
        return type(self).__name__

    @property
    @abstractmethod
    def suitability(self) -> list[AssetSuitability]:
        """Asset classes this strategy is appropriate for."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description of the strategy logic."""
