"""Tests for the US equity market-session clock."""

from __future__ import annotations

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from midas.market_hours import MARKET_TZ, is_market_open, next_session_open, session_close

ET = ZoneInfo("America/New_York")


def et(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=ET)


class TestIsMarketOpen:
    def test_open_mid_session_weekday(self) -> None:
        assert is_market_open(et(2026, 8, 18, 12, 0))  # Tuesday noon ET

    def test_closed_before_open_and_after_close(self) -> None:
        assert not is_market_open(et(2026, 8, 18, 9, 29))
        assert is_market_open(et(2026, 8, 18, 9, 30))  # inclusive open
        assert not is_market_open(et(2026, 8, 18, 16, 0))  # exclusive close
        assert is_market_open(et(2026, 8, 18, 15, 59))

    def test_closed_weekend(self) -> None:
        assert not is_market_open(et(2026, 8, 22, 12, 0))  # Saturday
        assert not is_market_open(et(2026, 8, 23, 12, 0))  # Sunday

    def test_closed_holiday(self) -> None:
        assert not is_market_open(et(2026, 11, 26, 12, 0))  # Thanksgiving

    def test_host_timezone_is_irrelevant(self) -> None:
        """The same instant must gate identically however it is expressed."""
        instant_utc = et(2026, 8, 18, 12, 0).astimezone(UTC)
        instant_tokyo = instant_utc.astimezone(ZoneInfo("Asia/Tokyo"))
        assert is_market_open(instant_utc)
        assert is_market_open(instant_tokyo)

    def test_dst_winter_and_summer_utc_offsets(self) -> None:
        """9:30 ET is 14:30 UTC under EST but 13:30 UTC under EDT."""
        # January (EST, UTC-5): 14:30 UTC == 9:30 ET.
        assert is_market_open(datetime(2026, 1, 20, 14, 30, tzinfo=UTC))
        assert not is_market_open(datetime(2026, 1, 20, 13, 30, tzinfo=UTC))
        # July (EDT, UTC-4): 13:30 UTC == 9:30 ET.
        assert is_market_open(datetime(2026, 7, 20, 13, 30, tzinfo=UTC))
        assert not is_market_open(datetime(2026, 7, 20, 12, 30, tzinfo=UTC))

    def test_unlisted_year_fails_open(self) -> None:
        """A stale holiday table must poll through, never sleep through."""
        assert is_market_open(et(2031, 1, 1, 12, 0))  # New Year's, but 2031 unlisted (Wednesday)


class TestNextSessionOpen:
    def test_saturday_sleeps_to_monday(self) -> None:
        nxt = next_session_open(et(2026, 8, 22, 12, 0)).astimezone(MARKET_TZ)
        assert nxt.date() == date(2026, 8, 24)
        assert (nxt.hour, nxt.minute) == (9, 30)

    def test_after_close_goes_to_next_day(self) -> None:
        nxt = next_session_open(et(2026, 8, 18, 16, 30)).astimezone(MARKET_TZ)
        assert nxt.date() == date(2026, 8, 19)

    def test_before_open_stays_same_day(self) -> None:
        nxt = next_session_open(et(2026, 8, 18, 7, 0)).astimezone(MARKET_TZ)
        assert nxt.date() == date(2026, 8, 18)

    def test_holiday_weekend_bridge(self) -> None:
        """Thu evening before the observed Jul 3 holiday skips to Mon Jul 6."""
        nxt = next_session_open(et(2026, 7, 2, 18, 0)).astimezone(MARKET_TZ)
        assert nxt.date() == date(2026, 7, 6)


class TestSessionClose:
    def test_close_is_1600_et(self) -> None:
        close = session_close(et(2026, 8, 18, 12, 0)).astimezone(MARKET_TZ)
        assert (close.hour, close.minute) == (16, 0)
        assert close.date() == date(2026, 8, 18)
