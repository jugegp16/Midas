"""Two-phase grid search optimizer for strategy parameters."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import pandas as pd
import yaml

from midas.agent import Agent
from midas.backtest import BacktestEngine
from midas.models import PortfolioConfig
from midas.sizing import SizingConfig, SizingEngine
from midas.strategies import STRATEGY_REGISTRY

# Default parameter ranges per strategy.
# Each entry: (min, max, coarse_step, fine_step)
PARAM_RANGES: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "MeanReversion": {
        "window": (10, 100, 10, 5),
        "threshold": (0.03, 0.25, 0.03, 0.01),
    },
    "ProfitTaking": {
        "gain_threshold": (0.10, 0.80, 0.10, 0.03),
    },
    "Momentum": {
        "window": (5, 50, 5, 2),
    },
}


@dataclass
class OptimizeResult:
    best_params: dict[str, dict[str, float]]
    best_return: float
    best_bh_return: float
    best_train_return: float
    best_test_return: float
    trials_run: int


def _build_grid(
    ranges: dict[str, tuple[float, float, float, float]],
    phase: str,
    center: dict[str, float] | None = None,
) -> list[dict[str, float]]:
    """Build a parameter grid for one strategy.

    Phase 'coarse' uses the full range with coarse steps.
    Phase 'fine' narrows around `center` with fine steps.
    """
    param_values: dict[str, list[float]] = {}
    for param, (lo, hi, coarse_step, fine_step) in ranges.items():
        if phase == "fine" and center is not None:
            c = center[param]
            step = fine_step
            lo_f = max(lo, c - coarse_step)
            hi_f = min(hi, c + coarse_step)
        else:
            lo_f, hi_f, step = lo, hi, coarse_step

        values = []
        v = lo_f
        while v <= hi_f + step * 0.01:  # small epsilon for float rounding
            values.append(round(v, 4))
            v += step
        param_values[param] = values

    # Cartesian product
    keys = list(param_values.keys())
    combos = list(itertools.product(*(param_values[k] for k in keys)))
    return [dict(zip(keys, combo, strict=False)) for combo in combos]


def _run_trial(
    strategy_params: dict[str, dict[str, float]],
    portfolio: PortfolioConfig,
    price_data: dict[str, pd.Series],
    start: date,
    end: date,
) -> tuple[float, float, float, float]:
    """Run a single backtest trial. Returns (total_return, bh_return,
    train_return, test_return)."""
    agents = []
    for name, params in strategy_params.items():
        cls = STRATEGY_REGISTRY[name]
        # Convert window params to int
        clean_params = {}
        for k, v in params.items():
            clean_params[k] = int(v) if k == "window" else v
        strategy = cls(**clean_params)
        agents.append(Agent(strategy=strategy, cooldown_days=5))

    sizing = SizingEngine(SizingConfig())
    engine = BacktestEngine(
        agents=agents, sizing_engine=sizing, enable_split=True
    )
    result = engine.run(portfolio, price_data, start, end)

    if result.starting_value <= 0:
        return 0.0, 0.0, 0.0, 0.0

    total_return = (
        (result.final_value - result.starting_value) / result.starting_value
    )
    bh_return = (
        (result.buy_and_hold_value - result.starting_value)
        / result.starting_value
    )
    return total_return, bh_return, result.train_return, result.test_return


def optimize(
    portfolio: PortfolioConfig,
    price_data: dict[str, pd.Series],
    start: date,
    end: date,
    strategy_names: list[str] | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> OptimizeResult:
    """Two-phase grid search over strategy parameters.

    Phase 1: coarse grid over full ranges.
    Phase 2: fine grid around the best result from phase 1.
    """
    log = log_fn or (lambda _: None)

    names = strategy_names or list(PARAM_RANGES.keys())
    # Filter to strategies that have defined ranges
    names = [n for n in names if n in PARAM_RANGES]

    if not names:
        msg = "No optimizable strategies found"
        raise ValueError(msg)

    # Phase 1: coarse grid
    log(f"Phase 1: coarse search over {', '.join(names)}...")
    coarse_grids = {
        name: _build_grid(PARAM_RANGES[name], "coarse") for name in names
    }

    # Total combinations = product of each strategy's grid size
    all_combos = list(
        itertools.product(*(coarse_grids[n] for n in names))
    )
    log(f"  {len(all_combos)} combinations to test")

    best_return = -float("inf")
    best_params: dict[str, dict[str, float]] = {}
    best_bh = 0.0
    best_train = 0.0
    best_test = 0.0

    for i, combo in enumerate(all_combos):
        params = dict(zip(names, combo, strict=False))
        total_ret, bh_ret, train_ret, test_ret = _run_trial(
            params, portfolio, price_data, start, end
        )
        if total_ret > best_return:
            best_return = total_ret
            best_params = params
            best_bh = bh_ret
            best_train = train_ret
            best_test = test_ret

        if (i + 1) % 100 == 0:
            log(f"  {i + 1}/{len(all_combos)} — best so far: {best_return:.2%}")

    log(f"  Phase 1 best: {best_return:.2%} with {best_params}")

    # Phase 2: fine grid around best
    log("Phase 2: fine search around best parameters...")
    fine_grids = {
        name: _build_grid(PARAM_RANGES[name], "fine", center=best_params[name])
        for name in names
    }

    fine_combos = list(
        itertools.product(*(fine_grids[n] for n in names))
    )
    log(f"  {len(fine_combos)} combinations to test")

    trials_total = len(all_combos)

    for i, combo in enumerate(fine_combos):
        params = dict(zip(names, combo, strict=False))
        total_ret, bh_ret, train_ret, test_ret = _run_trial(
            params, portfolio, price_data, start, end
        )
        if total_ret > best_return:
            best_return = total_ret
            best_params = params
            best_bh = bh_ret
            best_train = train_ret
            best_test = test_ret
        trials_total += 1

        if (i + 1) % 100 == 0:
            log(f"  {i + 1}/{len(fine_combos)} — best so far: {best_return:.2%}")

    log(f"  Phase 2 best: {best_return:.2%}")

    return OptimizeResult(
        best_params=best_params,
        best_return=round(best_return, 4),
        best_bh_return=round(best_bh, 4),
        best_train_return=round(best_train, 4),
        best_test_return=round(best_test, 4),
        trials_run=trials_total,
    )


def write_strategies_yaml(
    params: dict[str, dict[str, float]], path: str
) -> None:
    """Write optimized parameters to a strategies YAML file."""
    strategies = []
    for name, p in params.items():
        # Convert window to int for cleaner YAML
        clean = {}
        for k, v in p.items():
            clean[k] = int(v) if k == "window" else round(v, 4)
        strategies.append({"name": name, "params": clean})

    with open(path, "w") as f:
        yaml.dump({"strategies": strategies}, f, default_flow_style=False)
