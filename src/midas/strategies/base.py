"""Abstract strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from midas.models import AssetSuitability, StrategyTier


class Strategy(ABC):
    """Base class for all signal strategies.

    Strategies are stateless and ticker-agnostic. They receive price history
    and return a conviction score.
    """

    @abstractmethod
    def score(
        self,
        ticker: str,
        price_history: pd.Series,
        **kwargs: object,
    ) -> float | None:
        """Return conviction score for *ticker*.

        Positive = bullish, negative = bearish, 0 = neutral, None = abstain.
        Typically in [-1, +1].
        """

    @property
    def tier(self) -> StrategyTier:
        """Strategy tier — override in subclass if not CONVICTION."""
        return StrategyTier.CONVICTION

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        """Clamp *value* into [lo, hi]."""
        return max(lo, min(hi, value))

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name with parameters."""

    @property
    @abstractmethod
    def suitability(self) -> list[AssetSuitability]:
        """Asset classes this strategy is appropriate for."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description of the strategy logic."""
