"""Tests for the per-ticker price cache behind ``CachedYFinanceProvider``.

Every command (backtest, optimize, live) must see the same adjusted-price
series for a ticker no matter what window it asks for. Yahoo back-adjusts
all history for dividends and splits as of download time, so two separately
cached downloads of overlapping windows can disagree on the same bar. The
cache therefore keeps one series per ticker and replaces it wholesale.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

import midas.data.yfinance_provider as provider_module
from midas.data.yfinance_provider import CachedYFinanceProvider


class FakeYahoo:
    """Stand-in for ``yf.download`` that stamps each download with a vintage."""

    def __init__(self) -> None:
        self.calls: list[tuple[date, date]] = []
        self.vintage = 1.0

    def download(self, ticker: str, *, start: str, end: str, **_: object) -> pd.DataFrame:
        start_d = date.fromisoformat(start)
        end_exclusive = date.fromisoformat(end)
        self.calls.append((start_d, end_exclusive - timedelta(days=1)))
        days = pd.bdate_range(start_d, end_exclusive - timedelta(days=1))
        if len(days) == 0:
            return pd.DataFrame()
        price = pd.Series([100.0 * self.vintage] * len(days), index=days)
        return pd.DataFrame({"Open": price, "High": price, "Low": price, "Close": price, "Volume": price})


@pytest.fixture
def yahoo(monkeypatch: pytest.MonkeyPatch) -> FakeYahoo:
    fake = FakeYahoo()
    monkeypatch.setattr(provider_module, "yf", fake)
    return fake


def _provider(tmp_path: Path) -> CachedYFinanceProvider:
    return CachedYFinanceProvider(cache_dir=tmp_path / "cache")


def test_first_request_downloads_exactly_the_requested_window(tmp_path: Path, yahoo: FakeYahoo) -> None:
    frame = _provider(tmp_path).get_history("AAPL", date(2024, 1, 1), date(2024, 3, 29))
    assert yahoo.calls == [(date(2024, 1, 1), date(2024, 3, 29))]
    assert frame.index[0] == date(2024, 1, 1) and frame.index[-1] == date(2024, 3, 29)


def test_request_inside_cached_window_is_sliced_without_download(tmp_path: Path, yahoo: FakeYahoo) -> None:
    provider = _provider(tmp_path)
    provider.get_history("AAPL", date(2024, 1, 1), date(2024, 3, 29))
    frame = provider.get_history("AAPL", date(2024, 2, 1), date(2024, 2, 29))
    assert len(yahoo.calls) == 1
    assert frame.index[0] == date(2024, 2, 1) and frame.index[-1] == date(2024, 2, 29)


def test_request_outside_cached_window_downloads_the_union(tmp_path: Path, yahoo: FakeYahoo) -> None:
    provider = _provider(tmp_path)
    provider.get_history("AAPL", date(2024, 1, 1), date(2024, 3, 29))
    provider.get_history("AAPL", date(2023, 10, 2), date(2024, 2, 29))  # earlier start
    provider.get_history("AAPL", date(2024, 2, 1), date(2024, 6, 28))  # later end
    assert yahoo.calls[1] == (date(2023, 10, 2), date(2024, 3, 29))
    assert yahoo.calls[2] == (date(2023, 10, 2), date(2024, 6, 28))


def test_every_window_reads_the_latest_vintage(tmp_path: Path, yahoo: FakeYahoo) -> None:
    """The consistency guarantee: a refresh replaces the whole series, so an
    old window re-read afterwards sees the new adjustment, not its own stale copy."""
    provider = _provider(tmp_path)
    narrow = provider.get_history("AAPL", date(2024, 1, 1), date(2024, 3, 29))
    assert narrow["close"].iloc[0] == 100.0

    yahoo.vintage = 0.98  # a dividend went ex: Yahoo back-adjusts all history
    provider.get_history("AAPL", date(2023, 10, 2), date(2024, 3, 29))

    narrow_again = provider.get_history("AAPL", date(2024, 1, 1), date(2024, 3, 29))
    assert narrow_again["close"].iloc[0] == 98.0
    assert len(yahoo.calls) == 2


def test_cache_persists_across_provider_instances(tmp_path: Path, yahoo: FakeYahoo) -> None:
    _provider(tmp_path).get_history("AAPL", date(2024, 1, 1), date(2024, 3, 29))
    _provider(tmp_path).get_history("AAPL", date(2024, 1, 1), date(2024, 3, 29))
    assert len(yahoo.calls) == 1


def test_coverage_is_judged_by_requested_window_not_last_bar(tmp_path: Path, yahoo: FakeYahoo) -> None:
    """Live asks for ``end=today`` every tick; on a weekend the last bar is
    Friday. That must not trigger a re-download on every poll."""
    provider = _provider(tmp_path)
    saturday = date(2024, 3, 30)
    provider.get_history("AAPL", date(2024, 1, 1), saturday)
    provider.get_history("AAPL", date(2024, 1, 1), saturday)
    assert len(yahoo.calls) == 1


def test_tickers_are_cached_independently(tmp_path: Path, yahoo: FakeYahoo) -> None:
    provider = _provider(tmp_path)
    provider.get_history("AAPL", date(2024, 1, 1), date(2024, 3, 29))
    provider.get_history("MSFT", date(2024, 1, 1), date(2024, 3, 29))
    assert len(yahoo.calls) == 2


def test_empty_download_raises_and_caches_nothing(tmp_path: Path, yahoo: FakeYahoo) -> None:
    provider = _provider(tmp_path)
    with pytest.raises(ValueError, match="No data"):
        provider.get_history("AAPL", date(2024, 3, 30), date(2024, 3, 31))  # weekend only
    provider.get_history("AAPL", date(2024, 1, 1), date(2024, 3, 29))
    assert len(yahoo.calls) == 2
