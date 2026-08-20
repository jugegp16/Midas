"""US equity market-session clock for live-mode gating.

Regular sessions only: 9:30-16:00 ET, weekdays, minus full-closure
market holidays computed from the exchange's calendar rules (fixed
dates with weekend observation, nth-weekday floaters, Good Friday via
computus). Early closes (day after Thanksgiving, Christmas Eve) are
treated as full sessions — the cost is a couple of hours of harmless
polling on a handful of days per year. One-off special closures
(mourning days) are unknowable in advance and also poll harmlessly.

All decisions are made in America/New_York wall-clock time via zoneinfo,
so DST transitions are handled regardless of the host timezone.
"""

from __future__ import annotations

import functools
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)
MONDAY = 0
THURSDAY = 3


@functools.cache
def market_holidays(year: int) -> frozenset[date]:
    """Full-closure NYSE/Nasdaq holidays for *year*, computed from the rules.

    Fixed-date holidays observe weekend shifts (Saturday -> preceding
    Friday, Sunday -> following Monday), except New Year's Day, which the
    exchange does not observe early across the year boundary — a
    Saturday Jan 1 simply is not a market holiday. One-off special
    closures (e.g. presidential mourning days) cannot be computed and
    are absent; the engine polls harmlessly through them.
    """
    holidays = {
        _observed(date(year, 1, 1), friday_shift=False),  # New Year's Day
        _nth_weekday(year, 1, MONDAY, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, MONDAY, 3),  # Washington's Birthday
        _easter_sunday(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, MONDAY),  # Memorial Day
        _observed(date(year, 7, 4)),  # Independence Day
        _nth_weekday(year, 9, MONDAY, 1),  # Labor Day
        _nth_weekday(year, 11, THURSDAY, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),  # Christmas
    }
    if year >= 2022:  # market holiday since 2022
        holidays.add(_observed(date(year, 6, 19)))  # Juneteenth
    return frozenset(day for day in holidays if day is not None and day.year == year)


def _observed(holiday: date, friday_shift: bool = True) -> date | None:
    """Weekend-observation shift: Saturday -> Friday, Sunday -> Monday."""
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1) if friday_shift else None
    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)
    return holiday


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th *weekday* of *month* (n is 1-based)."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """The last *weekday* of *month*."""
    next_month = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    last = next_month - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday via the anonymous computus algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return date(year, month, day + 1)


def is_trading_day(day: date) -> bool:
    """Weekday and not a computed full-closure holiday."""
    return day.weekday() < 5 and day not in market_holidays(day.year)


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
