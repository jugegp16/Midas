"""Unit tests for the pure risk module. No allocator involvement here."""

from __future__ import annotations

import math

import numpy as np

from midas.risk import realized_vol


class TestRealizedVol:
    def test_constant_prices_yields_zero_vol(self) -> None:
        prices = np.full(100, 100.0)
        assert realized_vol(prices, lookback=60) == 0.0

    def test_known_synthetic_series(self) -> None:
        # Build a series with known daily log-return stdev = 0.01.
        rng = np.random.default_rng(0)
        daily_log_returns = rng.normal(loc=0.0, scale=0.01, size=10_000)
        prices = 100.0 * np.exp(np.cumsum(daily_log_returns))
        # Annualized: 0.01 * sqrt(252) ≈ 0.1587.
        vol = realized_vol(prices, lookback=prices.size - 1)
        assert math.isclose(vol, 0.01 * math.sqrt(252), rel_tol=0.05)

    def test_lookback_truncates(self) -> None:
        prices = np.array([100.0, 101.0, 100.0, 99.0, 100.0, 101.0, 102.0, 103.0])
        vol_3 = realized_vol(prices, lookback=3)
        vol_full = realized_vol(prices, lookback=len(prices) - 1)
        assert vol_3 != vol_full

    def test_insufficient_history_returns_zero(self) -> None:
        prices = np.array([100.0, 101.0])
        # Lookback bigger than available history → 0 (caller must check).
        assert realized_vol(prices, lookback=60) == 0.0
