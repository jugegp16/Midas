"""US equity market-session clock for live-mode gating.

Regular sessions only: 9:30-16:00 ET, weekdays, minus full-closure
market holidays. Early closes (day after Thanksgiving, Christmas Eve)
are treated as full sessions for now — the cost is a couple of hours of
harmless polling on a handful of days per year.

All decisions are made in America/New_York wall-clock time via zoneinfo,
so DST transitions are handled regardless of the host timezone.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

MARKET_TZ = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)

# Full-closure NYSE/Nasdaq holidays. Update yearly, like the tax brackets.
# Observed dates: a Saturday holiday closes the preceding Friday, a Sunday
# holiday the following Monday.
MARKET_HOLIDAYS: frozenset[date] = frozenset(
    {
        # 2026
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
        # 2027
        date(2027, 1, 1),  # New Year's Day
        date(2027, 1, 18),  # Martin Luther King Jr. Day
        date(2027, 2, 15),  # Washington's Birthday
        date(2027, 3, 26),  # Good Friday
        date(2027, 5, 31),  # Memorial Day
        date(2027, 6, 18),  # Juneteenth (observed; Jun 19 is a Saturday)
        date(2027, 7, 5),  # Independence Day (observed; Jul 4 is a Sunday)
        date(2027, 9, 6),  # Labor Day
        date(2027, 11, 25),  # Thanksgiving
        date(2027, 12, 24),  # Christmas (observed; Dec 25 is a Saturday)
    }
)
HOLIDAY_TABLE_YEARS = frozenset(day.year for day in MARKET_HOLIDAYS)

STALE_TABLE_YEARS_WARNED: set[int] = set()


def is_trading_day(day: date) -> bool:
    """Weekday and not a listed full-closure holiday.

    Fails OPEN when the holiday table has aged past *day*'s year: polling
    through an unlisted holiday is harmless (prices simply don't move),
    while failing closed would silently sleep through real sessions.
    """
    if day.weekday() >= 5:
        return False
    if day.year not in HOLIDAY_TABLE_YEARS and day.year not in STALE_TABLE_YEARS_WARNED:
        STALE_TABLE_YEARS_WARNED.add(day.year)
        logger.warning(
            "market holiday table has no entries for %d — update MARKET_HOLIDAYS in "
            "midas/market_hours.py; treating all weekdays as trading days until then",
            day.year,
        )
    return day not in MARKET_HOLIDAYS


def is_market_open(now: datetime) -> bool:
    """Whether the regular US equity session is open at *now* (tz-aware)."""
    local = now.astimezone(MARKET_TZ)
    if not is_trading_day(local.date()):
        return False
    return SESSION_OPEN <= local.time() < SESSION_CLOSE


def next_session_open(now: datetime) -> datetime:
    """The next session open at or after *now*, as a UTC datetime.

    If *now* is inside a session, returns that session's open (in the
    past); callers gate with :func:`is_market_open` first.
    """
    local = now.astimezone(MARKET_TZ)
    day = local.date()
    same_day_open_possible = is_trading_day(day) and local.time() < SESSION_CLOSE
    candidate = day if same_day_open_possible else day + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    open_local = datetime.combine(candidate, SESSION_OPEN, tzinfo=MARKET_TZ)
    return open_local.astimezone(UTC)


def session_close(now: datetime) -> datetime:
    """The close of *now*'s trading day in UTC (valid when a session is open)."""
    local = now.astimezone(MARKET_TZ)
    close_local = datetime.combine(local.date(), SESSION_CLOSE, tzinfo=MARKET_TZ)
    return close_local.astimezone(UTC)
