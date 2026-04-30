"""Tests for the Optuna-based optimizer."""

import math
from datetime import date

import pandas as pd
import pytest
from conftest import make_price_series

from midas.metrics import compute_annualized_return
from midas.models import Holding, PortfolioConfig, RiskConfig
from midas.optimizer import (
    PARAM_RANGES,
    OptimizeResult,
    WalkForwardResult,
    optimize,
    walk_forward_optimize,
    write_strategies_yaml,
)


def _make_optimizer_data() -> tuple[PortfolioConfig, dict[str, pd.DataFrame], date, date]:
    """Create a portfolio and price data suitable for optimisation tests."""
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="TEST", shares=10, cost_basis=100.0)],
        available_cash=2000.0,
    )
    # Price drops then recovers — mean reversion should find something.
    returns = [0.0] * 20 + [-0.008] * 20 + [0.01] * 30 + [0.0] * 30
    prices = make_price_series(date(2024, 1, 2), 100, 100.0, returns, name="TEST")
    start = min(prices.index)
    end = max(prices.index)
    return portfolio, {"TEST": prices}, start, end


def test_optimize_returns_result() -> None:
    """optimize() should return a populated OptimizeResult."""
    portfolio, price_data, start, end = _make_optimizer_data()
    result = optimize(
        portfolio=portfolio,
        price_data=price_data,
        start=start,
        end=end,
        strategy_names=["MeanReversion"],
        n_trials=5,
    )
    assert isinstance(result, OptimizeResult)
    assert result.trials_run == 5
    assert "MeanReversion" in result.best_params
    # Check that all expected params are present
    for param in PARAM_RANGES["MeanReversion"]:
        assert param in result.best_params["MeanReversion"]


def test_optimize_multiple_strategies() -> None:
    """optimize() should handle multiple strategies simultaneously."""
    portfolio, price_data, start, end = _make_optimizer_data()
    result = optimize(
        portfolio=portfolio,
        price_data=price_data,
        start=start,
        end=end,
        strategy_names=["MeanReversion", "Momentum"],
        n_trials=5,
    )
    assert "MeanReversion" in result.best_params
    assert "Momentum" in result.best_params


def test_optimize_rejects_no_strategies() -> None:
    """optimize() should raise when no optimisable strategies are found."""
    portfolio, price_data, start, end = _make_optimizer_data()
    with pytest.raises(ValueError, match="No optimizable strategies"):
        optimize(
            portfolio=portfolio,
            price_data=price_data,
            start=start,
            end=end,
            strategy_names=["NotAStrategy"],
            n_trials=5,
        )


def test_optimize_log_fn_called() -> None:
    """The log callback should be invoked during optimisation."""
    portfolio, price_data, start, end = _make_optimizer_data()
    messages: list[str] = []
    optimize(
        portfolio=portfolio,
        price_data=price_data,
        start=start,
        end=end,
        strategy_names=["MeanReversion"],
        n_trials=5,
        log_fn=messages.append,
    )
    assert len(messages) >= 2  # at least start + finish messages


def test_write_strategies_yaml(tmp_path) -> None:
    """write_strategies_yaml should produce valid YAML with correct structure."""
    import yaml

    params = {
        "MeanReversion": {"window": 20.0, "threshold": 0.1, "_weight": 1.5},
        "TrailingStop": {"trail_pct": 0.1},
    }
    out = str(tmp_path / "strats.yaml")
    write_strategies_yaml(params, out)

    with open(out) as f:
        data = yaml.safe_load(f)

    assert "strategies" in data
    names = [s["name"] for s in data["strategies"]]
    assert "MeanReversion" in names
    assert "TrailingStop" in names

    mr = next(s for s in data["strategies"] if s["name"] == "MeanReversion")
    assert mr["weight"] == 1.5
    assert mr["params"]["window"] == 20

    ts = next(s for s in data["strategies"] if s["name"] == "TrailingStop")
    assert ts["params"]["trail_pct"] == 0.1
    assert "weight" not in ts


def test_write_strategies_yaml_omits_risk_block_when_default(tmp_path) -> None:
    """An all-default RiskConfig produces no ``risk:`` key — same as no input block."""
    import yaml

    params = {"MeanReversion": {"window": 20.0, "threshold": 0.1}}
    out = str(tmp_path / "strats.yaml")
    write_strategies_yaml(params, out, risk_config=RiskConfig())

    with open(out) as f:
        data = yaml.safe_load(f)
    assert "risk" not in data


def test_write_strategies_yaml_omits_risk_block_when_none(tmp_path) -> None:
    """``risk_config=None`` (no input config) → no ``risk:`` block, current behavior."""
    import yaml

    params = {"MeanReversion": {"window": 20.0, "threshold": 0.1}}
    out = str(tmp_path / "strats.yaml")
    write_strategies_yaml(params, out, risk_config=None)

    with open(out) as f:
        data = yaml.safe_load(f)
    assert "risk" not in data


def test_write_strategies_yaml_preserves_user_risk_block(tmp_path) -> None:
    """A user-supplied risk policy round-trips through the optimized YAML output.

    Regression: the optimizer historically dropped any ``risk:`` block from the
    input, so users who configured CPPI / vol target / inverse-vol weighting
    silently lost it on optimize. The block must round-trip — only non-default
    fields are emitted, mirroring the spec's "omit to disable" YAML conventions.
    """
    import yaml

    params = {"MeanReversion": {"window": 20.0, "threshold": 0.1}}
    risk = RiskConfig(
        weighting="inverse_vol",
        vol_lookback_days=90,
        vol_target=0.20,
        drawdown_penalty=1.5,
        drawdown_floor=0.5,
    )
    out = str(tmp_path / "strats.yaml")
    write_strategies_yaml(params, out, risk_config=risk)

    with open(out) as f:
        data = yaml.safe_load(f)
    assert data["risk"] == {
        "weighting": "inverse_vol",
        "vol_lookback_days": 90,
        "vol_target": 0.20,
        "drawdown_penalty": 1.5,
        "drawdown_floor": 0.5,
    }


def test_write_strategies_yaml_emits_only_nondefault_risk_fields(tmp_path) -> None:
    """Defaults are omitted so the output mirrors what the user typed."""
    import yaml

    params = {"MeanReversion": {"window": 20.0, "threshold": 0.1}}
    risk = RiskConfig(vol_target=0.18)  # everything else default
    out = str(tmp_path / "strats.yaml")
    write_strategies_yaml(params, out, risk_config=risk)

    with open(out) as f:
        data = yaml.safe_load(f)
    assert data["risk"] == {"vol_target": 0.18}


# ---------------------------------------------------------------------------
# Walk-forward tests
# ---------------------------------------------------------------------------


def _make_walk_forward_data() -> tuple[PortfolioConfig, dict[str, pd.DataFrame], date, date]:
    """Longer price series suitable for walk-forward folds.

    400 days: 60% train = 240 days, remaining 160 days → 2 folds of ~63 days.
    """
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="TEST", shares=10, cost_basis=100.0)],
        available_cash=2000.0,
    )
    returns = [0.005] * 100 + [-0.003] * 100 + [0.004] * 100 + [0.002] * 100
    prices = make_price_series(date(2023, 1, 2), 400, 100.0, returns, name="TEST")
    start = min(prices.index)
    end = max(prices.index)
    return portfolio, {"TEST": prices}, start, end


def test_walk_forward_returns_result() -> None:
    """walk_forward_optimize() should return a WalkForwardResult with per-fold data."""
    portfolio, price_data, start, end = _make_walk_forward_data()
    result = walk_forward_optimize(
        portfolio=portfolio,
        price_data=price_data,
        start=start,
        end=end,
        strategy_names=["MeanReversion"],
        n_trials=10,
    )
    assert isinstance(result, WalkForwardResult)
    assert len(result.folds) >= 2
    assert "MeanReversion" in result.best_params

    # Summary metrics should be populated.
    assert 0 <= result.winning_folds <= len(result.folds)
    assert result.worst_fold_return <= max(f.test_return for f in result.folds)
    assert result.efficiency_ratio != float("inf")
    # CAGR should be nonzero when folds have nonzero returns.
    assert result.annualized_return != 0.0


def test_walk_forward_folds_are_sequential() -> None:
    """Each fold's test window should follow its training window without overlap."""
    portfolio, price_data, start, end = _make_walk_forward_data()
    result = walk_forward_optimize(
        portfolio=portfolio,
        price_data=price_data,
        start=start,
        end=end,
        strategy_names=["MeanReversion"],
        n_trials=10,
    )
    for f in result.folds:
        assert f.train_start < f.train_end
        assert f.train_end < f.test_start
        assert f.test_start <= f.test_end

    # Training windows are anchored (all start at the same date) and grow.
    for i in range(1, len(result.folds)):
        assert result.folds[i].train_start == result.folds[0].train_start
        assert result.folds[i].train_end > result.folds[i - 1].train_end


def test_walk_forward_per_fold_returns_are_annualized() -> None:
    """Each fold's stored return must equal its raw return, annualized.

    Guards against a regression where `train_return`/`test_return` silently
    hold the cumulative value again (what this PR was meant to fix).
    """
    portfolio, price_data, start, end = _make_walk_forward_data()
    result = walk_forward_optimize(
        portfolio=portfolio,
        price_data=price_data,
        start=start,
        end=end,
        strategy_names=["MeanReversion"],
        n_trials=10,
    )
    for fold in result.folds:
        train_days = (fold.train_end - fold.train_start).days
        test_days = (fold.test_end - fold.test_start).days
        expected_train = round(compute_annualized_return(fold.train_return_raw, train_days), 4)
        expected_test = round(compute_annualized_return(fold.test_return_raw, test_days), 4)
        assert fold.train_return == expected_train
        assert fold.test_return == expected_test


def test_walk_forward_cagr_compounds_raw_returns_not_annualized() -> None:
    """Overall OOS CAGR must compound raw per-fold returns, not already-annualized ones.

    This is the double-annualization trap the PR explicitly avoids. If a
    future refactor swapped `test_return_raw` for `test_return` in the
    compounding loop, the overall CAGR would be wildly inflated on short
    folds — this test pins the correct semantics.
    """
    portfolio, price_data, start, end = _make_walk_forward_data()
    result = walk_forward_optimize(
        portfolio=portfolio,
        price_data=price_data,
        start=start,
        end=end,
        strategy_names=["MeanReversion"],
        n_trials=10,
    )

    compounded = 1.0
    for fold in result.folds:
        compounded *= 1.0 + fold.test_return_raw
    first_test_start = result.folds[0].test_start
    last_test_end = result.folds[-1].test_end
    years = (last_test_end - first_test_start).days / 365.25
    expected = compounded ** (1.0 / years) - 1.0 if compounded > 0 and years > 0 else 0.0
    assert math.isclose(result.annualized_return, round(expected, 4), abs_tol=1e-4)

    # Sanity: the double-annualized value would be computed from the
    # already-annualized per-fold returns and should differ from the
    # correct value (otherwise the test is vacuous).
    double_compounded = 1.0
    for fold in result.folds:
        double_compounded *= 1.0 + fold.test_return
    double_annualized = double_compounded ** (1.0 / years) - 1.0 if double_compounded > 0 and years > 0 else 0.0
    assert not math.isclose(double_annualized, expected, abs_tol=1e-4)


def test_walk_forward_efficiency_ratio_uses_annualized_returns() -> None:
    """Efficiency ratio must be mean(annualized_test) / mean(annualized_train)."""
    portfolio, price_data, start, end = _make_walk_forward_data()
    result = walk_forward_optimize(
        portfolio=portfolio,
        price_data=price_data,
        start=start,
        end=end,
        strategy_names=["MeanReversion"],
        n_trials=10,
    )
    mean_test = sum(fold.test_return for fold in result.folds) / len(result.folds)
    mean_train = sum(fold.train_return for fold in result.folds) / len(result.folds)
    if mean_train == 0:
        pytest.skip("Degenerate fixture: no train-side return to compare against")
    expected = round(mean_test / mean_train, 4)
    assert result.efficiency_ratio == expected


def test_walk_forward_too_few_days() -> None:
    """walk_forward_optimize() should raise when data is too short."""
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="TEST", shares=10, cost_basis=100.0)],
        available_cash=2000.0,
    )
    prices = make_price_series(date(2024, 1, 2), 50, 100.0, name="TEST")
    start, end = min(prices.index), max(prices.index)
    with pytest.raises(ValueError, match="Not enough data"):
        walk_forward_optimize(
            portfolio=portfolio,
            price_data={"TEST": prices},
            start=start,
            end=end,
            strategy_names=["MeanReversion"],
            n_trials=10,
        )
