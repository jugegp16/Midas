"""YFinance data provider with simple file-based caching."""

from __future__ import annotations

import hashlib
import pickle
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from midas.data.provider import DataProvider

_DEFAULT_CACHE_DIR = Path.home() / ".midas_cache"


class CachedYFinanceProvider(DataProvider):
    def __init__(self, cache_dir: Path = _DEFAULT_CACHE_DIR) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get_history(self, ticker: str, start: date, end: date) -> pd.Series:
        cache_key = self._cache_path(ticker, start, end)
        if cache_key.exists():
            with open(cache_key, "rb") as f:
                return pickle.load(f)

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

        series: pd.Series = df["Close"].squeeze()
        series.index = pd.to_datetime(series.index).date  # type: ignore[assignment]
        series.name = ticker

        with open(cache_key, "wb") as f:
            pickle.dump(series, f)

        return series

    def get_current_price(self, ticker: str) -> float:
        t = yf.Ticker(ticker)
        hist = t.history(period="1d")
        if hist.empty:
            msg = f"No current price available for {ticker}"
            raise ValueError(msg)
        return float(hist["Close"].iloc[-1])

    def _cache_path(self, ticker: str, start: date, end: date) -> Path:
        key = f"{ticker}_{start}_{end}"
        hashed = hashlib.md5(key.encode()).hexdigest()
        return self._cache_dir / f"{hashed}.pkl"
