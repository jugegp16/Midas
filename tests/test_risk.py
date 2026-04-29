"""Unit tests for the pure risk module. No allocator involvement here."""

from __future__ import annotations

import math

import numpy as np

from midas.risk import covariance_matrix, realized_vol


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


class TestCovarianceMatrix:
    def test_shape_and_symmetry(self) -> None:
        rng = np.random.default_rng(0)
        log_returns = rng.normal(0, 0.01, size=(252, 3))
        cov = covariance_matrix(log_returns)
        assert cov.shape == (3, 3)
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_positive_semidefinite(self) -> None:
        rng = np.random.default_rng(0)
        log_returns = rng.normal(0, 0.01, size=(252, 4))
        cov = covariance_matrix(log_returns)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert (eigenvalues >= -1e-12).all()

    def test_diagonal_dominant_for_independent_series(self) -> None:
        rng = np.random.default_rng(0)
        n = 5_000
        log_returns = np.column_stack(
            [
                rng.normal(0, 0.005, n),
                rng.normal(0, 0.01, n),
                rng.normal(0, 0.02, n),
            ],
        )
        cov = covariance_matrix(log_returns)
        # Diagonal scales with per-series vol; off-diagonals stay small.
        assert cov[0, 0] < cov[1, 1] < cov[2, 2]
        assert abs(cov[0, 1]) < cov[0, 0]
        assert abs(cov[0, 2]) < cov[0, 0]
