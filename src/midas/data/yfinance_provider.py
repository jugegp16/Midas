"""YFinance data provider with a per-ticker on-disk cache.

One cached series per ticker, replaced wholesale on every refresh. Yahoo
back-adjusts all history for dividends and splits as of download time, and
the adjusted values also drift by ~1e-4 with the requested start date, so
two separately cached downloads of overlapping windows disagree on the same
bar — and backtest, optimize, and live each ask for a different window
(warmup buffers, ``end=today``). A 1e-4 price difference flips an integer
share count on day one and compounds into a visibly different backtest.
Keying the cache by ticker alone means every command reads the same series
at any moment, and a refresh triggered by one command (typically live
extending ``end`` to today) refreshes it for all. Results are reproducible
against a given cache file, not across refreshes.
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf  # type: ignore[import-untyped]

from midas.data.provider import DataProvider

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".midas_cache"


@dataclass(frozen=True)
class CachedSeries:
    """One ticker's cached history plus the window it was requested for.

    Coverage is judged by the *requested* window, not the last bar: live asks
    for ``end=today`` every tick, and on a weekend the last bar is Friday —
    that must not re-download on every poll.
    """

    start: date
    end: date
    frame: pd.DataFrame

    def covers(self, start: date, end: date) -> bool:
        return self.start <= start and self.end >= end


class CachedYFinanceProvider(DataProvider):
    """Yahoo Finance provider with one cached series per ticker.

    A request inside the cached window is served as a slice. A request
    outside it downloads the union of the two windows and atomically
    replaces the ticker's file, so the whole series moves to the new
    adjustment vintage together.
    """

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get_history(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        cached = self._load(ticker)
        if cached is None or not cached.covers(start, end):
            fetch_start = start if cached is None else min(start, cached.start)
            fetch_end = end if cached is None else max(end, cached.end)
            cached = CachedSeries(fetch_start, fetch_end, self._download(ticker, fetch_start, fetch_end))
            self._store(ticker, cached)
        frame = cached.frame
        return frame[(frame.index >= start) & (frame.index <= end)]

    def _download(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Fetch ``[start, end]`` from Yahoo and normalise to the OHLCV frame.

        Raises:
            ValueError: When Yahoo returns no bars for the window.
        """
        # yfinance end is exclusive, so add a day
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            msg = f"No data returned for {ticker} between {start} and {end}"
            raise ValueError(msg)

        # yfinance can return a MultiIndex on columns when multiple tickers
        # are requested — guard against a single-ticker MultiIndex too.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        volume = df["Volume"].astype(float)
        # yfinance can emit NaN volume on indexes, illiquid crypto, or
        # pre-market-adjusted bars. VWAPReversion's rolling sum is poisoned
        # by NaN, so coerce to zero — those bars then fall back to a simple
        # mean of typical price. Log the count so a chronic data-quality
        # issue (e.g. an index with no real volume) is visible instead of
        # silently degrading the strategy.
        nan_volume_count = int(volume.isna().sum())
        if nan_volume_count > 0:
            logger.warning(
                "%s: %d bar(s) with NaN volume between %s and %s — coerced to 0; "
                "VWAP-weighted strategies will fall back to typical-price SMA on those bars.",
                ticker,
                nan_volume_count,
                start,
                end,
            )
            volume = volume.fillna(0.0)

        frame = pd.DataFrame(
            {
                "open": df["Open"].astype(float),
                "high": df["High"].astype(float),
                "low": df["Low"].astype(float),
                "close": df["Close"].astype(float),
                "volume": volume,
            }
        )
        frame.index = pd.to_datetime(frame.index).date
        frame.index.name = "date"
        return frame

    def get_current_price(self, ticker: str) -> float:
        yf_ticker = yf.Ticker(ticker)
        hist = yf_ticker.history(period="1d")
        if hist.empty:
            msg = f"No current price available for {ticker}"
            raise ValueError(msg)
        return float(hist["Close"].iloc[-1])

    def _cache_path(self, ticker: str) -> Path:
        return self._cache_dir / f"{ticker}.pkl"

    def _load(self, ticker: str) -> CachedSeries | None:
        path = self._cache_path(ticker)
        if not path.exists():
            return None
        with open(path, "rb") as handle:
            cached = pickle.load(handle)
        if not isinstance(cached, CachedSeries):
            # Not one of ours (foreign or corrupt file) — ignore and refresh.
            return None
        return cached

    def _store(self, ticker: str, cached: CachedSeries) -> None:
        """Write via a temp file and rename so a concurrent reader never sees a partial pickle."""
        path = self._cache_path(ticker)
        tmp_path = path.with_suffix(".pkl.tmp")
        with open(tmp_path, "wb") as handle:
            pickle.dump(cached, handle)
        os.replace(tmp_path, path)
