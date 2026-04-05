"""Abstract strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from midas.models import AssetSuitability, MechanicalIntent, StrategyTier


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
