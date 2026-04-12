"""PriceHistory: the OHLCV bar struct threaded through strategies and the allocator.

All strategies receive a ``PriceHistory`` rather than a bare close array so
that high/low/open/volume-dependent indicators (ATR, true range, Donchian,
VWAP, gap detection) can read what they need without out-of-band state.

Backed by numpy arrays — not pandas — so the hot path (per-day precompute
and allocator lookups) stays allocation-free. Providers return pandas
DataFrames at the boundary; the backtest and live engines convert once
into ``PriceHistory`` and then thread the struct through the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PriceHistory:
    """OHLCV bars for a single ticker over some date range.

    ``volume`` is optional because not every provider has it (some crypto
    feeds, OTC quotes). Strategies that require volume must check and
    raise a clear error when it is missing.

    Slicing (``hist[:n]``) returns a new ``PriceHistory`` with each
    underlying array sliced identically. ``len(hist)`` returns the number
    of bars. Both operations are O(1) over numpy views — no copies.
    """

    dates: np.ndarray  # object array of datetime.date
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.close.shape[0])

    def __getitem__(self, key: slice) -> PriceHistory:
        if not isinstance(key, slice):
            msg = f"PriceHistory only supports slice indexing, got {type(key).__name__}"
            raise TypeError(msg)
        return PriceHistory(
            dates=self.dates[key],
            open=self.open[key],
            high=self.high[key],
            low=self.low[key],
            close=self.close[key],
            volume=self.volume[key] if self.volume is not None else None,
        )

    @property
    def last_date(self) -> date | None:
        if len(self) == 0:
            return None
        return self.dates[-1]  # type: ignore[no-any-return]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> PriceHistory:
        """Build a ``PriceHistory`` from a provider-style OHLCV DataFrame.

        Required columns: ``open``, ``high``, ``low``, ``close``. ``volume``
        is optional. The DataFrame index must be a sequence of ``date``
        objects (provider contract). Values are copied into numpy arrays.
        """
        required = ("open", "high", "low", "close")
        missing = [col for col in required if col not in df.columns]
        if missing:
            msg = f"PriceHistory requires columns {required}; missing: {missing}"
            raise ValueError(msg)
        return cls(
            dates=np.asarray(df.index, dtype=object),
            open=np.asarray(df["open"].values, dtype=float),
            high=np.asarray(df["high"].values, dtype=float),
            low=np.asarray(df["low"].values, dtype=float),
            close=np.asarray(df["close"].values, dtype=float),
            volume=np.asarray(df["volume"].values, dtype=float) if "volume" in df.columns else None,
        )

    @classmethod
    def from_close_only(
        cls,
        dates: np.ndarray,
        close: np.ndarray,
    ) -> PriceHistory:
        """Build a ``PriceHistory`` from close-only data by synthesizing OHLV.

        Used by tests and legacy code paths that only have closes. The
        synthesized open/high/low all equal close and volume is ``None``.
        Strategies that look at highs/lows will get degenerate output —
        fine for tests that only exercise close-based logic.
        """
        close_arr = np.asarray(close, dtype=float)
        return cls(
            dates=np.asarray(dates, dtype=object),
            open=close_arr.copy(),
            high=close_arr.copy(),
            low=close_arr.copy(),
            close=close_arr,
            volume=None,
        )
