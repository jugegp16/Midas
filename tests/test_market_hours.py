"""Tests for the US equity market-session clock."""

from __future__ import annotations

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from midas.market_hours import MARKET_TZ, is_market_open, market_holidays, next_session_open, session_close

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

    def test_future_years_need_no_maintenance(self) -> None:
        """The rule-based calendar closes holidays in any year."""
        assert not is_market_open(et(2031, 1, 1, 12, 0))  # New Year's 2031 (Wednesday)
        assert not is_market_open(et(2030, 11, 28, 12, 0))  # Thanksgiving 2030


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


class TestMarketHolidays:
    def test_matches_hand_verified_2026_2027_calendar(self) -> None:
        """Ground truth: the independently hand-checked NYSE closure lists."""
        assert market_holidays(2026) == frozenset(
            {
                date(2026, 1, 1),  # New Year's Day
                date(2026, 1, 19),  # Martin Luther King Jr. Day
                date(2026, 2, 16),  # Washington's Birthday
                date(2026, 4, 3),  # Good Friday
                date(2026, 5, 25),  # Memorial Day
                date(2026, 6, 19),  # Juneteenth
                date(2026, 7, 3),  # Independence Day (observed; Jul 4 is a Saturday)
                date(2026, 9, 7),  # Labor Day
                date(2026, 11, 26),  # Thanksgiving
                date(2026, 12, 25),  # Christmas
            }
        )
        assert market_holidays(2027) == frozenset(
            {
                date(2027, 1, 1),
                date(2027, 1, 18),
                date(2027, 2, 15),
                date(2027, 3, 26),  # Good Friday (Easter 2027 is Mar 28)
                date(2027, 5, 31),
                date(2027, 6, 18),  # Juneteenth observed (Jun 19 is a Saturday)
                date(2027, 7, 5),  # Independence Day observed (Jul 4 is a Sunday)
                date(2027, 9, 6),
                date(2027, 11, 25),
                date(2027, 12, 24),  # Christmas observed (Dec 25 is a Saturday)
            }
        )

    def test_saturday_new_year_is_not_observed_early(self) -> None:
        """Jan 1 2022 fell on a Saturday: NYSE did NOT close Dec 31 2021."""
        assert date(2021, 12, 31) not in market_holidays(2021)
        assert date(2022, 1, 1) not in market_holidays(2022)

    def test_juneteenth_only_from_2022(self) -> None:
        assert date(2021, 6, 18) not in market_holidays(2021)  # Jun 19 2021 Sat, not yet a holiday
        assert date(2023, 6, 19) in market_holidays(2023)

    def test_good_friday_tracks_easter(self) -> None:
        assert date(2025, 4, 18) in market_holidays(2025)  # Easter 2025 is Apr 20
        assert date(2024, 3, 29) in market_holidays(2024)  # Easter 2024 is Mar 31
