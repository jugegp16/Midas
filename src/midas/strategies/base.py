"""Strategy base classes.

Two distinct strategy roles, enforced by the type system:

- ``EntrySignal`` produces a [0, 1] bullish score per ticker. Scores are
  blended by the Allocator into target weights via softmax. Entries are
  the only thing that drives buys.

- ``ExitRule`` evaluates the lots of an open position and emits direct
  ``ExitIntent`` objects (one per triggered lot, analogous to "sell lot N because Y").
  Exits never participate in score blending and never go through the
  target-weight path; the OrderSizer converts intents to sized sell orders.

Both inherit from ``Strategy`` for shared bookkeeping (``name``,
``warmup_period``, ``suitability``, ``description``, ``tier_label``), but
the two scoring interfaces (``EntrySignal.score`` and
``ExitRule.evaluate_exit``) are completely disjoint.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import ClassVar

import numpy as np

from midas.models import AssetSuitability, ExitIntent, PositionLot

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
    """Minimal base for all strategies. Subclass ``EntrySignal`` or ``ExitRule``."""

    tier_label: ClassVar[str] = "Strategy"

    @property
    def warmup_period(self) -> int:
        """Bars of price history required before this strategy produces valid output.

        Backtest/optimize/live use ``max(warmup_period)`` across configured
        strategies to prefetch a lookback buffer before the user's start date,
        so strategies can emit valid signals from day one of the simulation
        rather than spending the first N days in warmup.

        Default is 0. Override in subclasses that depend on a rolling window.
        For recursive indicators (EMA/RSI/MACD) multiply the nominal period by
        ``RECURSIVE_WARMUP_MULTIPLIER`` so the indicator has room to converge
        to its steady-state value.
        """
        return 0

    @staticmethod
    def clamp(value: float, lo: float, hi: float) -> float:
        """Clamp *value* into [lo, hi]."""
        return max(lo, min(hi, value))

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


class EntrySignal(Strategy):
    """Strategy that produces a [0, 1] bullish entry score.

    Scores are blended by the Allocator via softmax to produce target
    portfolio weights. A score of 0 means "no opinion" — the strategy
    contributes nothing to the budget for this ticker. None means
    "abstain" — the strategy can't score this ticker yet (insufficient
    history, missing data, etc.).
    """

    tier_label: ClassVar[str] = "Entry Signal"

    @abstractmethod
    def score(
        self,
        price_history: np.ndarray,
        **kwargs: object,
    ) -> float | None:
        """Return entry score in ``[0, 1]`` (or ``None`` to abstain)."""

    def precompute(self, prices: np.ndarray) -> np.ndarray | None:
        """Precompute scores for every prefix of *prices* in one pass.

        Returns an array *s* of length ``len(prices)`` where ``s[i]`` equals
        ``self.score(prices[:i+1])`` (or ``NaN`` when ``score`` would return
        ``None``). Returns ``None`` when precomputation is not possible.
        """
        return None


class ExitRule(Strategy):
    """Strategy that emits direct exit intents based on lot state.

    Receives the FIFO lot list for one open position plus the current price
    history and returns zero or more ``ExitIntent`` objects (one per
    triggered lot) describing which lot to sell and why. Exit intents bypass
    the target-weight blend entirely — the OrderSizer dedupes per-lot
    intents across all rules and converts them into sell Orders.
    """

    tier_label: ClassVar[str] = "Exit Rule"

    @abstractmethod
    def evaluate_exit(
        self,
        ticker: str,
        lots: list[PositionLot],
        price_history: np.ndarray,
    ) -> list[ExitIntent]:
        """Return a list of exit intents for *ticker* given its open *lots*."""

    # ----- shared helpers used by lot-aware and full-position exit rules -----

    def fire_on_lots(
        self,
        ticker: str,
        lots: list[PositionLot],
        current: float,
        predicate: Callable[[PositionLot], bool],
        reason: Callable[[PositionLot], str],
    ) -> list[ExitIntent]:
        """Build one ExitIntent per lot matching ``predicate``.

        Lots with non-positive shares or non-positive cost basis are skipped.
        ``reason(lot)`` produces the human-readable log string for a single
        triggered lot. Returns an empty list when no lot triggers.
        """
        intents: list[ExitIntent] = []
        for i, lot in enumerate(lots):
            if lot.cost_basis <= 0 or lot.shares <= 0:
                continue
            if predicate(lot):
                intents.append(
                    ExitIntent(
                        ticker=ticker,
                        lot_index=i,
                        lot_shares=lot.shares,
                        source=self.name,
                        reason=reason(lot),
                    )
                )
        return intents

    def sell_all(
        self,
        ticker: str,
        lots: list[PositionLot],
        current: float,
        reason: str,
    ) -> list[ExitIntent]:
        """Build one ExitIntent per lot to liquidate the entire open position."""
        return [
            ExitIntent(
                ticker=ticker,
                lot_index=i,
                lot_shares=lot.shares,
                source=self.name,
                reason=reason,
            )
            for i, lot in enumerate(lots)
            if lot.shares > 0
        ]
