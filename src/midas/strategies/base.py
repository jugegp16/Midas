"""Abstract strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime

import pandas as pd

from midas.models import AssetSuitability, Direction, Signal


class Strategy(ABC):
    """Base class for all signal strategies.

    Strategies are stateless and ticker-agnostic. They receive price history
    and return zero or more signals.
    """

    @abstractmethod
    def evaluate(
        self,
        ticker: str,
        price_history: pd.Series,
        **kwargs: object,
    ) -> list[Signal]:
        """Evaluate the strategy against a ticker's price history.

        Strategies that need extra context (e.g. cost_basis) declare those
        as explicit keyword arguments in their override.  Unknown kwargs
        are silently absorbed so callers can pass a uniform context dict.
        """

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

    def _make_signal(
        self,
        ticker: str,
        direction: Direction,
        strength: float,
        reasoning: str,
        price: float,
    ) -> Signal:
        return Signal(
            ticker=ticker,
            direction=direction,
            strength=round(min(1.0, max(0.0, strength)), 2),
            reasoning=reasoning,
            timestamp=datetime.now(tz=UTC),
            price=price,
            strategy_name=self.name,
        )
