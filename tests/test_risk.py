"""Tests for the risk-discipline module."""

from __future__ import annotations

import math

import numpy as np
from conftest import ph, ph_ohlc  # noqa: F401 — ph_ohlc reserved for future tests

from midas.risk import realized_vol


class TestRealizedVol:
    def test_constant_series_returns_zero(self):
        history = ph(np.full(100, 100.0))
        vol = realized_vol(history, window=30)
        assert vol == 0.0

    def test_known_series_matches_hand_computed(self):
        # Daily returns: exactly +1% and -1% alternating over 30 bars (29 returns).
        # Each daily return is ±0.01 (computed from ratio p_i / p_{i-1}).
        prices = [100.0]
        for i in range(1, 31):
            prices.append(prices[-1] * (1.01 if i % 2 == 1 else 1 / 1.01))
        history = ph(np.asarray(prices))

        vol = realized_vol(history, window=30)
        # daily std of ±0.01 (equal-sized up/down moves) ~= 0.01; annualized * sqrt(252)
        assert vol is not None
        assert math.isclose(vol, 0.01 * math.sqrt(252), rel_tol=2e-2)

    def test_insufficient_history_returns_none(self):
        history = ph(np.full(10, 100.0))
        assert realized_vol(history, window=30) is None

    def test_annualize_false_returns_daily_vol(self):
        prices = [100.0]
        for i in range(1, 31):
            prices.append(prices[-1] * (1.01 if i % 2 == 1 else 1 / 1.01))
        history = ph(np.asarray(prices))
        daily = realized_vol(history, window=30, annualize=False)
        annual = realized_vol(history, window=30, annualize=True)
        assert daily is not None and annual is not None
        assert math.isclose(annual / daily, math.sqrt(252), rel_tol=1e-6)

    def test_nan_in_window_returns_none(self):
        prices = np.full(60, 100.0)
        prices[-5] = float("nan")
        history = ph(prices)
        # Window includes the NaN → insufficient clean history.
        assert realized_vol(history, window=30) is None

    def test_exact_window_size_is_sufficient(self):
        # Window == len(history) should succeed (not fail with off-by-one).
        history = ph(np.full(30, 100.0))
        assert realized_vol(history, window=30) == 0.0
