"""YFinance data provider with simple file-based caching."""

from __future__ import annotations

import hashlib
import logging
import pickle
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf  # type: ignore[import-untyped]

from midas.data.provider import DataProvider

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".midas_cache"


class CachedYFinanceProvider(DataProvider):
    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get_history(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        cache_key = self._cache_path(ticker, start, end)
        if cache_key.exists():
            with open(cache_key, "rb") as f:
                return pickle.load(f)  # type: ignore[no-any-return]

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

        with open(cache_key, "wb") as f:
            pickle.dump(frame, f)

        return frame

    def get_current_price(self, ticker: str) -> float:
        yf_ticker = yf.Ticker(ticker)
        hist = yf_ticker.history(period="1d")
        if hist.empty:
            msg = f"No current price available for {ticker}"
            raise ValueError(msg)
        return float(hist["Close"].iloc[-1])

    def _cache_path(self, ticker: str, start: date, end: date) -> Path:
        key = f"{ticker}_{start}_{end}_ohlcv"
        hashed = hashlib.md5(key.encode()).hexdigest()
        return self._cache_dir / f"{hashed}.pkl"
