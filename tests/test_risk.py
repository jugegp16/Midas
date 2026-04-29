"""Unit tests for the pure risk module. No allocator involvement here."""

from __future__ import annotations

import math

import numpy as np

from midas.models import DEFAULT_VOL_FLOOR
from midas.risk import (
    apply_drawdown_overlay,
    covariance_matrix,
    inverse_vol_offset,
    predict_portfolio_vol,
    realized_vol,
)


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


class TestPredictPortfolioVol:
    def test_uncorrelated_equal_weight_diversification(self) -> None:
        # Three independent series, daily stdev 0.01 each → annualized ≈ 0.1587.
        # Equal weight across 3 → predicted ≈ 0.1587 / sqrt(3) ≈ 0.0916.
        n = 20_000
        rng = np.random.default_rng(0)
        log_returns = np.column_stack(
            [
                rng.normal(0, 0.01, n),
                rng.normal(0, 0.01, n),
                rng.normal(0, 0.01, n),
            ],
        )
        weights = np.array([1 / 3, 1 / 3, 1 / 3])
        predicted = predict_portfolio_vol(weights, log_returns)
        assert math.isclose(predicted, 0.01 * math.sqrt(252) / math.sqrt(3), rel_tol=0.10)

    def test_perfectly_correlated_no_diversification(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(0, 0.01, 5_000)
        log_returns = np.column_stack([x, x])
        weights = np.array([0.5, 0.5])
        predicted = predict_portfolio_vol(weights, log_returns)
        # Same as either ticker's vol — no diversification.
        assert math.isclose(predicted, 0.01 * math.sqrt(252), rel_tol=0.05)

    def test_zero_weights_yields_zero(self) -> None:
        rng = np.random.default_rng(0)
        log_returns = rng.normal(0, 0.01, size=(100, 3))
        weights = np.array([0.0, 0.0, 0.0])
        assert predict_portfolio_vol(weights, log_returns) == 0.0


class TestApplyDrawdownOverlay:
    def test_no_drawdown_no_change(self) -> None:
        scale = apply_drawdown_overlay(current_drawdown=0.0, penalty=1.5, floor=0.5)
        assert scale == 1.0

    def test_moderate_drawdown(self) -> None:
        # 20% DD * 1.5 penalty = 0.30 reduction → 0.70 exposure.
        scale = apply_drawdown_overlay(current_drawdown=0.20, penalty=1.5, floor=0.5)
        assert math.isclose(scale, 0.70)

    def test_floor_binds_at_deep_drawdown(self) -> None:
        # 50% DD * 1.5 penalty = 0.75 reduction → 0.25 raw, floored at 0.5.
        scale = apply_drawdown_overlay(current_drawdown=0.50, penalty=1.5, floor=0.5)
        assert scale == 0.5

    def test_penalty_zero_disables(self) -> None:
        scale = apply_drawdown_overlay(current_drawdown=0.30, penalty=0.0, floor=0.5)
        assert scale == 1.0

    def test_negative_drawdown_clamps_at_one(self) -> None:
        # Defensive: a negative current_drawdown can only result from a driver
        # bug. Spec demands exposure_scale saturate at 1.0, never above.
        scale = apply_drawdown_overlay(current_drawdown=-0.10, penalty=1.5, floor=0.5)
        assert scale == 1.0


class TestInverseVolOffset:
    def test_offset_is_negative_log_vol(self) -> None:
        assert math.isclose(inverse_vol_offset(0.20, vol_floor=DEFAULT_VOL_FLOOR), -math.log(0.20))
        assert math.isclose(inverse_vol_offset(0.50, vol_floor=DEFAULT_VOL_FLOOR), -math.log(0.50))

    def test_floor_clamps_low_vols(self) -> None:
        # Floor is a log(0) guard well below any realistic annualized vol;
        # use an explicit floor parameter to exercise the clamp behavior.
        assert math.isclose(
            inverse_vol_offset(0.001, vol_floor=0.005),
            -math.log(0.005),
        )

    def test_default_floor_does_not_bind_for_normal_vols(self) -> None:
        # A 0.5% annualized vol is unusually quiet but realistic for some
        # ETFs; the default floor must not bind there.
        assert math.isclose(
            inverse_vol_offset(0.005, vol_floor=DEFAULT_VOL_FLOOR),
            -math.log(0.005),
        )

    def test_zero_vol_returns_nan(self) -> None:
        # Caller must check; zero indicates insufficient signal.
        assert math.isnan(inverse_vol_offset(0.0, vol_floor=DEFAULT_VOL_FLOOR))
