"""Tests for the risk-discipline module."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from conftest import ph, ph_ohlc  # noqa: F401 — ph_ohlc reserved for future tests

from midas.risk import (
    apply_instrument_diversification_multiplier,
    apply_vol_targeting,
    covariance_matrix,
    predict_portfolio_vol,
    realized_vol,
)


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


def _prices_from_returns(daily_returns: np.ndarray, base: float = 100.0) -> np.ndarray:
    """Build a close array from daily returns."""
    out = np.empty(len(daily_returns) + 1)
    out[0] = base
    for i, ret in enumerate(daily_returns):
        out[i + 1] = out[i] * (1.0 + ret)
    return out


class TestCovarianceMatrix:
    def test_uncorrelated_series_give_diagonal_correlation(self):
        rng = np.random.default_rng(42)
        # 260 bars each — enough for 252-day corr window + slack.
        rets_a = rng.normal(0.0, 0.01, 260)
        rets_b = rng.normal(0.0, 0.02, 260)
        hist_a = ph(_prices_from_returns(rets_a))
        hist_b = ph(_prices_from_returns(rets_b))

        cov = covariance_matrix(
            {"A": hist_a, "B": hist_b},
            vol_window=60,
            corr_window=252,
            vol_floor=0.001,
        )

        assert cov is not None
        # Diagonals should be ~ (realized vol ** 2)
        # Off-diagonal correlation should be near zero.
        corr_ab = cov.loc["A", "B"] / math.sqrt(cov.loc["A", "A"] * cov.loc["B", "B"])
        assert abs(corr_ab) < 0.2  # shrinkage may pull slightly toward 0 or away

    def test_perfectly_correlated_series_give_unit_correlation(self):
        rng = np.random.default_rng(7)
        base_rets = rng.normal(0.0, 0.01, 260)
        # B is identical to A — scaled vol by 2x to make the math nontrivial.
        hist_a = ph(_prices_from_returns(base_rets))
        hist_b = ph(_prices_from_returns(base_rets * 2.0))

        cov = covariance_matrix(
            {"A": hist_a, "B": hist_b},
            vol_window=60,
            corr_window=252,
            vol_floor=0.001,
        )

        assert cov is not None
        corr_ab = cov.loc["A", "B"] / math.sqrt(cov.loc["A", "A"] * cov.loc["B", "B"])
        # LedoitWolf shrinks correlation toward the mean off-diagonal, which
        # for a 2-asset system with one off-diagonal = 1.0 still stays near 1.
        assert corr_ab > 0.9

    def test_vol_floor_clamps_low_vol_ticker(self):
        rng = np.random.default_rng(3)
        rets_normal = rng.normal(0.0, 0.01, 260)
        rets_tiny = rng.normal(0.0, 0.0001, 260)  # well below floor
        hist_normal = ph(_prices_from_returns(rets_normal))
        hist_tiny = ph(_prices_from_returns(rets_tiny))

        cov = covariance_matrix(
            {"N": hist_normal, "T": hist_tiny},
            vol_window=60,
            corr_window=252,
            vol_floor=0.05,  # 5% annualized
        )

        assert cov is not None
        vol_t = math.sqrt(cov.loc["T", "T"])
        # Vol floor is annualized 5% → variance ≈ 0.0025 on the diagonal.
        assert vol_t >= 0.05 - 1e-9

    def test_insufficient_history_returns_none(self):
        short = ph(np.full(50, 100.0))  # < corr_window of 252
        cov = covariance_matrix(
            {"A": short, "B": short},
            vol_window=30,
            corr_window=252,
            vol_floor=0.001,
        )
        assert cov is None

    def test_output_is_psd(self):
        rng = np.random.default_rng(99)
        hists = {name: ph(_prices_from_returns(rng.normal(0, 0.01, 260))) for name in ("A", "B", "C", "D")}
        cov = covariance_matrix(hists, vol_window=60, corr_window=252, vol_floor=0.001)
        assert cov is not None
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert (eigenvalues >= -1e-9).all()

    def test_index_and_columns_match_ticker_order(self):
        rng = np.random.default_rng(1)
        hists = {name: ph(_prices_from_returns(rng.normal(0, 0.01, 260))) for name in ("Z", "A", "M")}
        cov = covariance_matrix(hists, vol_window=60, corr_window=252, vol_floor=0.001)
        assert cov is not None
        # Deterministic order for downstream matmul correctness.
        assert list(cov.index) == ["A", "M", "Z"]
        assert list(cov.columns) == ["A", "M", "Z"]


def _corr_frame(tickers: list[str], matrix: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=tickers, columns=tickers)


class TestApplyInstrumentDiversificationMultiplier:
    def test_uncorrelated_equal_weights_give_idm_approx_sqrt_n(self):
        tickers = ["A", "B", "C", "D"]
        weights = {t: 0.25 for t in tickers}
        corr = _corr_frame(tickers, np.eye(4))
        out = apply_instrument_diversification_multiplier(weights, corr, cap=5.0)
        # IDM = 1/sqrt(0.25 * 4 * 0.25) = 1/sqrt(0.25) = 2.0 = sqrt(4).
        total = sum(out.values())
        assert math.isclose(total, sum(weights.values()) * 2.0, rel_tol=1e-9)

    def test_perfect_correlation_gives_idm_one(self):
        tickers = ["A", "B"]
        weights = {"A": 0.4, "B": 0.4}
        corr = _corr_frame(tickers, np.ones((2, 2)))
        out = apply_instrument_diversification_multiplier(weights, corr, cap=5.0)
        assert math.isclose(out["A"], 0.4, rel_tol=1e-9)
        assert math.isclose(out["B"], 0.4, rel_tol=1e-9)

    def test_cap_binds_when_raw_idm_exceeds(self):
        tickers = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        weights = dict.fromkeys(tickers, 1.0 / 9.0)
        corr = _corr_frame(tickers, np.eye(9))
        # raw IDM = sqrt(9) = 3.0; cap at 2.0 → multiplier = 2.0 / 3.0 of raw, so scale = 2.0.
        out = apply_instrument_diversification_multiplier(weights, corr, cap=2.0)
        total = sum(out.values())
        assert math.isclose(total, sum(weights.values()) * 2.0, rel_tol=1e-9)

    def test_empty_weights_returns_empty(self):
        corr = _corr_frame(["A"], np.eye(1))
        assert apply_instrument_diversification_multiplier({}, corr, cap=2.5) == {}

    def test_single_ticker_weights_unchanged(self):
        weights = {"A": 0.5}
        corr = _corr_frame(["A"], np.eye(1))
        out = apply_instrument_diversification_multiplier(weights, corr, cap=2.5)
        assert out == {"A": 0.5}

    def test_zero_weights_returns_unchanged(self):
        tickers = ["A", "B"]
        weights = dict.fromkeys(tickers, 0.0)
        corr = _corr_frame(tickers, np.eye(2))
        out = apply_instrument_diversification_multiplier(weights, corr, cap=2.5)
        assert out == weights

    def test_weights_referencing_missing_ticker_are_ignored(self):
        # "A" is in weights but absent from the correlation matrix — treat as
        # uncorrelated (contributes to sum but not to the diversification math).
        weights = {"A": 0.3, "B": 0.3}
        corr = _corr_frame(["B"], np.eye(1))
        out = apply_instrument_diversification_multiplier(weights, corr, cap=2.5)
        # A passes through with multiplier 1.0; B scales by its own IDM which
        # for a 1-asset "portfolio" is trivially 1.0.
        assert math.isclose(out["A"], 0.3, rel_tol=1e-9)
        assert math.isclose(out["B"], 0.3, rel_tol=1e-9)


def _cov_frame(tickers: list[str], matrix: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=tickers, columns=tickers)


class TestApplyVolTargeting:
    def test_below_target_returns_unchanged(self):
        # diag(cov) = 0.01 (10% ann vol), w = 0.5,0.5 uncorrelated → portfolio vol ≈ 7.07%
        tickers = ["A", "B"]
        cov = _cov_frame(tickers, np.diag([0.01, 0.01]))
        weights = {"A": 0.5, "B": 0.5}
        out = apply_vol_targeting(weights, cov, target_annualized_vol=0.20)
        assert out == weights

    def test_exactly_at_target_returns_unchanged(self):
        # Single asset, vol ≈ 20%; weight = 1 → portfolio vol = 20%.
        tickers = ["A"]
        cov = _cov_frame(tickers, np.array([[0.04]]))
        weights = {"A": 1.0}
        out = apply_vol_targeting(weights, cov, target_annualized_vol=0.20)
        assert math.isclose(out["A"], 1.0, rel_tol=1e-9)

    def test_double_target_halves_weights(self):
        # Single asset, vol = 40% (variance 0.16), target = 20% → scale by 0.5.
        tickers = ["A"]
        cov = _cov_frame(tickers, np.array([[0.16]]))
        weights = {"A": 1.0}
        out = apply_vol_targeting(weights, cov, target_annualized_vol=0.20)
        assert math.isclose(out["A"], 0.5, rel_tol=1e-9)

    def test_zero_weights_returns_zero(self):
        tickers = ["A", "B"]
        cov = _cov_frame(tickers, np.eye(2))
        weights = {"A": 0.0, "B": 0.0}
        out = apply_vol_targeting(weights, cov, target_annualized_vol=0.20)
        assert out == weights

    def test_degenerate_cov_returns_unchanged(self):
        tickers = ["A", "B"]
        cov = _cov_frame(tickers, np.zeros((2, 2)))
        weights = {"A": 0.5, "B": 0.5}
        out = apply_vol_targeting(weights, cov, target_annualized_vol=0.20)
        assert out == weights

    def test_weights_missing_from_cov_pass_through_unscaled(self):
        # "B" isn't in cov. "A" dominates the vol calculation.
        tickers = ["A"]
        cov = _cov_frame(tickers, np.array([[0.16]]))  # 40% vol
        weights = {"A": 1.0, "B": 0.3}
        out = apply_vol_targeting(weights, cov, target_annualized_vol=0.20)
        # A scales by 0.5; B is uncovered → passes through scaled by the same
        # portfolio-wide multiplier so the downscaling remains coherent.
        assert math.isclose(out["A"], 0.5, rel_tol=1e-9)
        assert math.isclose(out["B"], 0.15, rel_tol=1e-9)


class TestPredictPortfolioVol:
    def test_two_asset_uncorrelated_equal_weight(self):
        # var_a = var_b = 0.04 (20% vol), w = (0.5, 0.5), uncorrelated.
        # portfolio_var = 0.25 * 0.04 + 0.25 * 0.04 = 0.02 → vol ≈ sqrt(0.02) ≈ 0.1414
        tickers = ["A", "B"]
        cov = _cov_frame(tickers, np.diag([0.04, 0.04]))
        vol = predict_portfolio_vol({"A": 0.5, "B": 0.5}, cov)
        assert math.isclose(vol, math.sqrt(0.02), rel_tol=1e-9)

    def test_zero_weights_return_zero(self):
        tickers = ["A"]
        cov = _cov_frame(tickers, np.array([[0.04]]))
        assert predict_portfolio_vol({"A": 0.0}, cov) == 0.0

    def test_empty_weights_return_zero(self):
        tickers = ["A"]
        cov = _cov_frame(tickers, np.array([[0.04]]))
        assert predict_portfolio_vol({}, cov) == 0.0

    def test_weights_outside_cov_index_ignored(self):
        tickers = ["A"]
        cov = _cov_frame(tickers, np.array([[0.04]]))
        # Only A contributes; B is ignored.
        vol = predict_portfolio_vol({"A": 1.0, "B": 5.0}, cov)
        assert math.isclose(vol, 0.20, rel_tol=1e-9)

    def test_degenerate_cov_returns_zero(self):
        tickers = ["A", "B"]
        cov = _cov_frame(tickers, -np.eye(2))  # non-PSD, forces quad <= 0
        assert predict_portfolio_vol({"A": 1.0, "B": 1.0}, cov) == 0.0
