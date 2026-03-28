"""Tests for the Optuna-based optimizer."""

from datetime import date

import pandas as pd
import pytest

from midas.models import Holding, PortfolioConfig
from midas.optimizer import (
    PARAM_RANGES,
    OptimizeResult,
    optimize,
    write_strategies_yaml,
)
from tests.conftest import make_price_series


def _make_optimizer_data() -> tuple[PortfolioConfig, dict[str, pd.Series], date, date]:
    """Create a portfolio and price data suitable for optimisation tests."""
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="TEST", shares=10, cost_basis=100.0)],
        available_cash=2000.0,
    )
    # Price drops then recovers — mean reversion should find something.
    returns = (
        [0.0] * 20
        + [-0.008] * 20
        + [0.01] * 30
        + [0.0] * 30
    )
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
