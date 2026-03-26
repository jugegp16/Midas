"""Tests for trading restriction enforcement."""

from __future__ import annotations

from datetime import date

from midas.models import Direction, TradingRestrictions
from midas.restrictions import RestrictionTracker


class TestRestrictionTracker:
    def test_no_restriction_when_disabled(self) -> None:
        tracker = RestrictionTracker(TradingRestrictions(round_trip_days=0))
        tracker.record_trade("AAPL", Direction.BUY, date(2024, 1, 1))
        assert not tracker.is_blocked("AAPL", Direction.SELL, date(2024, 1, 2))

    def test_sell_blocked_within_window(self) -> None:
        tracker = RestrictionTracker(TradingRestrictions(round_trip_days=30))
        tracker.record_trade("AAPL", Direction.BUY, date(2024, 1, 1))
        assert tracker.is_blocked("AAPL", Direction.SELL, date(2024, 1, 15))

    def test_sell_allowed_after_window(self) -> None:
        tracker = RestrictionTracker(TradingRestrictions(round_trip_days=30))
        tracker.record_trade("AAPL", Direction.BUY, date(2024, 1, 1))
        assert not tracker.is_blocked("AAPL", Direction.SELL, date(2024, 2, 1))

    def test_buy_blocked_after_sell_within_window(self) -> None:
        tracker = RestrictionTracker(TradingRestrictions(round_trip_days=30))
        tracker.record_trade("AAPL", Direction.SELL, date(2024, 1, 1))
        assert tracker.is_blocked("AAPL", Direction.BUY, date(2024, 1, 15))

    def test_same_direction_not_blocked(self) -> None:
        tracker = RestrictionTracker(TradingRestrictions(round_trip_days=30))
        tracker.record_trade("AAPL", Direction.BUY, date(2024, 1, 1))
        assert not tracker.is_blocked("AAPL", Direction.BUY, date(2024, 1, 2))

    def test_different_tickers_independent(self) -> None:
        tracker = RestrictionTracker(TradingRestrictions(round_trip_days=30))
        tracker.record_trade("AAPL", Direction.BUY, date(2024, 1, 1))
        assert not tracker.is_blocked("MSFT", Direction.SELL, date(2024, 1, 2))

    def test_no_trade_history_not_blocked(self) -> None:
        tracker = RestrictionTracker(TradingRestrictions(round_trip_days=30))
        assert not tracker.is_blocked("AAPL", Direction.BUY, date(2024, 1, 1))

    def test_exact_boundary_day_allowed(self) -> None:
        tracker = RestrictionTracker(TradingRestrictions(round_trip_days=30))
        tracker.record_trade("AAPL", Direction.BUY, date(2024, 1, 1))
        # Day 30 exactly should be allowed
        assert not tracker.is_blocked("AAPL", Direction.SELL, date(2024, 1, 31))

    def test_day_before_boundary_blocked(self) -> None:
        tracker = RestrictionTracker(TradingRestrictions(round_trip_days=30))
        tracker.record_trade("AAPL", Direction.BUY, date(2024, 1, 1))
        # Day 29 should still be blocked
        assert tracker.is_blocked("AAPL", Direction.SELL, date(2024, 1, 30))
