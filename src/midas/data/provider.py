"""Abstract data provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class DataProvider(ABC):
    """Contract for market data providers.

    Providers return OHLCV bars as a ``pd.DataFrame`` indexed by
    ``datetime.date``. Required columns: ``open``, ``high``, ``low``,
    ``close``. ``volume`` is optional (not every feed has it). Close
    prices are assumed adjusted for splits and dividends.

    The backtest and live engines convert these DataFrames into
    :class:`midas.data.price_history.PriceHistory` structs once at the
    boundary so downstream code (allocator, strategies) operates on
    numpy arrays rather than pandas objects.
    """

    @abstractmethod
    def get_history(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Return OHLCV bars for *ticker* between *start* and *end* inclusive."""

    @abstractmethod
    def get_current_price(self, ticker: str) -> float:
        """Return the latest available price for a ticker."""
