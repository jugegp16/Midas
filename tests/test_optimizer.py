"""Tests for the Optuna-based optimizer."""

from datetime import date

import pandas as pd
import pytest
from conftest import make_price_series

from midas.models import Holding, PortfolioConfig
from midas.optimizer import (
    PARAM_RANGES,
    OptimizeResult,
    WalkForwardResult,
    optimize,
    walk_forward_optimize,
    write_strategies_yaml,
)


def _make_optimizer_data() -> tuple[PortfolioConfig, dict[str, pd.Series], date, date]:
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
            strategy_names=["DollarCostAveraging"],
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
        "TrailingStop": {"trail_pct": 0.1, "_veto_threshold": -0.5},
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
    assert ts["veto_threshold"] == -0.5


# ---------------------------------------------------------------------------
# Walk-forward tests
# ---------------------------------------------------------------------------


def _make_walk_forward_data() -> tuple[PortfolioConfig, dict[str, pd.Series], date, date]:
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
