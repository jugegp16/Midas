"""Regression: the optimizer silently dropped ``forecast_scaling``.

Same failure mode as the dropped ``risk:`` block: policy knobs from the
strategies YAML must reach every optimizer trial (else parameters are
tuned under a different blending regime than the one they'll run with)
and must round-trip into the optimized output YAML (else the next run
silently reverts to raw blending).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from conftest import make_price_frame

from midas.config import load_strategies
from midas.models import Holding, PortfolioConfig
from midas.optimizer import ALLOCATION_KEY, _run_trial, write_strategies_yaml


def _three_ticker_portfolio() -> tuple[PortfolioConfig, dict[str, pd.DataFrame]]:
    portfolio = PortfolioConfig(
        holdings=[
            Holding(ticker="AAA", shares=10, cost_basis=100.0),
            Holding(ticker="BBB", shares=10, cost_basis=100.0),
            Holding(ticker="CCC", shares=10, cost_basis=100.0),
        ],
        available_cash=3_000.0,
    )
    price_data: dict[str, pd.DataFrame] = {}
    for seed, ticker in enumerate(("AAA", "BBB", "CCC"), start=1):
        rng = np.random.default_rng(seed)
        returns = list(rng.normal(0.0, 0.02, 200))
        price_data[ticker] = make_price_frame(
            start=date(2024, 1, 1), days=200, base_price=100.0, daily_returns=returns, name=ticker
        )
    return portfolio, price_data


def test_forecast_scaling_changes_trial_results() -> None:
    """Behavioral: quantile vs none must produce different trial outcomes.

    Two rules with different score distributions over volatile data, with
    the position cap lifted above the budget so blend weights actually
    drive sizing. If both modes produce identical results, the knob
    plumbed through to the constraints but the trial ignored it.
    """
    portfolio, price_data = _three_ticker_portfolio()
    strategy_params = {
        ALLOCATION_KEY: {"max_position_pct": 0.6},
        "MeanReversion": {"window": 20, "threshold": 0.02, "_weight": 1.0},
        "RSIOversold": {"window": 14, "oversold_threshold": 40.0, "_weight": 1.0},
    }

    final_values: dict[str, float] = {}
    for mode in ("none", "quantile"):
        *_, result = _run_trial(
            strategy_params,
            portfolio,
            price_data,
            start=date(2024, 1, 1),
            end=date(2024, 7, 1),
            enable_split=False,
            forecast_scaling=mode,
        )
        final_values[mode] = result.final_value

    assert final_values["none"] != final_values["quantile"]


def test_write_strategies_yaml_round_trips_forecast_scaling(tmp_path: Path) -> None:
    """The optimized YAML must preserve the user's forecast_scaling policy."""
    params = {
        ALLOCATION_KEY: {"max_position_pct": 0.3},
        "MeanReversion": {"window": 20, "threshold": 0.05, "_weight": 1.0},
    }
    out = tmp_path / "optimized.yaml"
    write_strategies_yaml(params, str(out), forecast_scaling="quantile")

    _, constraints, _ = load_strategies(out)
    assert constraints.forecast_scaling == "quantile"


def test_write_strategies_yaml_omits_default_forecast_scaling(tmp_path: Path) -> None:
    """Default none is not emitted, and absent means none on reload."""
    params = {
        ALLOCATION_KEY: {"max_position_pct": 0.3},
        "MeanReversion": {"window": 20, "threshold": 0.05, "_weight": 1.0},
    }
    out = tmp_path / "optimized.yaml"
    write_strategies_yaml(params, str(out))

    assert "forecast_scaling" not in out.read_text()
    _, constraints, _ = load_strategies(out)
    assert constraints.forecast_scaling == "none"
