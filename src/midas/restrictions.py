"""Trading restriction enforcement — blocks trades that violate configurable rules."""

from __future__ import annotations

from datetime import date

from midas.models import Direction, TradingRestrictions


class RestrictionTracker:
    """Tracks executed trades and enforces round-trip restrictions.

    Round-trip restriction: after buying (selling) a ticker, you cannot sell
    (buy) it until `round_trip_days` have elapsed.
    """

    def __init__(self, restrictions: TradingRestrictions) -> None:
        self._restrictions = restrictions
        # (ticker, direction) -> date of last execution
        self._last_trade: dict[tuple[str, Direction], date] = {}

    def is_blocked(self, ticker: str, direction: Direction, today: date) -> bool:
        """Return True if executing this trade would violate a restriction."""
        if self._restrictions.round_trip_days <= 0:
            return False

        opposite = Direction.SELL if direction == Direction.BUY else Direction.BUY
        last = self._last_trade.get((ticker, opposite))
        if last is None:
            return False

        days_since = (today - last).days
        return days_since < self._restrictions.round_trip_days

    def record_trade(self, ticker: str, direction: Direction, today: date) -> None:
        """Record an executed trade for future restriction checks."""
        self._last_trade[(ticker, direction)] = today
