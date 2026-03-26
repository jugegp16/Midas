"""Abstract data provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class DataProvider(ABC):
    """Contract for market data providers.

    All providers return adjusted close (total return) data as a pd.Series
    indexed by date.
    """

    @abstractmethod
    def get_history(self, ticker: str, start: date, end: date) -> pd.Series:
        """Return adjusted close prices for ticker between start and end (inclusive)."""

    @abstractmethod
    def get_current_price(self, ticker: str) -> float:
        """Return the latest available price for a ticker."""
