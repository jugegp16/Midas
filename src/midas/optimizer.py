"""Two-phase grid search optimizer for strategy parameters."""

from __future__ import annotations

import itertools
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date

import pandas as pd
import yaml

from midas.allocator import Allocator
from midas.backtest import BacktestEngine
from midas.models import AllocationConstraints, PortfolioConfig, StrategyTier
from midas.rebalancer import Rebalancer
from midas.strategies import STRATEGY_REGISTRY

# Parameters that should be cast to int when building strategy instances.
_INT_PARAMS = {
    "window", "short_window", "long_window",
    "fast_period", "slow_period", "signal_period",
    "frequency_days",
}

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
    "RSIOversold": {
        "window": (7, 28, 7, 2),
        "oversold_threshold": (15.0, 40.0, 5.0, 2.0),
    },
    "RSIOverbought": {
        "window": (7, 28, 7, 2),
        "overbought_threshold": (60.0, 85.0, 5.0, 2.0),
    },
    "BollingerBand": {
        "window": (10, 50, 10, 5),
        "num_std": (1.5, 3.0, 0.5, 0.25),
    },
    "MACDCrossover": {
        "fast_period": (8, 16, 4, 2),
        "slow_period": (20, 32, 4, 2),
        "signal_period": (5, 13, 4, 2),
    },
    "DollarCostAveraging": {
        "frequency_days": (5, 30, 5, 2),
    },
    "GapDownRecovery": {
        "gap_threshold": (0.02, 0.08, 0.02, 0.005),
    },
    "TrailingStop": {
        "trail_pct": (0.05, 0.25, 0.05, 0.02),
    },
    "StopLoss": {
        "loss_threshold": (0.05, 0.25, 0.05, 0.02),
    },
    "VWAPReversion": {
        "window": (10, 50, 10, 5),
        "threshold": (0.01, 0.05, 0.01, 0.005),
    },
    "MovingAverageCrossover": {
        "short_window": (10, 30, 5, 2),
        "long_window": (40, 100, 10, 5),
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
    """Run a single backtest trial with the allocator+rebalancer system.

    Returns (total_return, bh_return, train_return, test_return).
    """
    conviction: list[tuple] = []
    protective: list[tuple] = []
    mechanical = []

    for name, params in strategy_params.items():
        cls = STRATEGY_REGISTRY[name]
        clean_params = {
            k: int(v) if k in _INT_PARAMS else v for k, v in params.items()
        }
        strategy = cls(**clean_params)

        if strategy.tier == StrategyTier.PROTECTIVE:
            protective.append((strategy, -0.5))  # default veto threshold
        elif strategy.tier == StrategyTier.MECHANICAL:
            mechanical.append(strategy)
        else:
            conviction.append((strategy, 1.0))  # uniform weight

    # Count tickers in portfolio
    n_tickers = sum(1 for h in portfolio.holdings if h.shares > 0)
    constraints = AllocationConstraints()

    allocator = Allocator(conviction, protective, constraints, n_tickers)
    rebalancer = Rebalancer()

    engine = BacktestEngine(
        allocator=allocator,
        rebalancer=rebalancer,
        mechanical_strategies=mechanical,
        constraints=constraints,
        enable_split=True,
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


_worker_state: dict = {}


def _init_worker(portfolio, price_data, start, end):
    _worker_state.update(
        portfolio=portfolio, price_data=price_data, start=start, end=end
    )


def _trial_worker(args):
    names, combo = args
    params = dict(zip(names, combo, strict=False))
    return (*_run_trial(params, **_worker_state), params)


def _run_parallel(combos, names, portfolio, price_data, start, end, log):
    """Run all combos across worker processes, return best result."""
    max_workers = min((os.cpu_count() or 4) // 2, len(combos)) or 1
    log(f"  {len(combos)} combinations to test ({max_workers} workers)")

    best = (-float("inf"), {}, 0.0, 0.0, 0.0)
    done = 0
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(portfolio, price_data, start, end),
    ) as pool:
        chunksize = max(1, len(combos) // (max_workers * 4))
        for *scores, params in pool.map(
            _trial_worker, [(names, c) for c in combos], chunksize=chunksize
        ):
            done += 1
            if scores[0] > best[0]:
                best = (scores[0], params, scores[1], scores[2], scores[3])
            if done % 500 == 0:
                log(f"  {done}/{len(combos)} — best so far: {best[0]:.2%}")

    return best


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
    # Filter to strategies that have defined ranges and are not MECHANICAL
    names = [
        n for n in names
        if n in PARAM_RANGES
        and STRATEGY_REGISTRY[n]().tier != StrategyTier.MECHANICAL
    ]

    if not names:
        msg = "No optimizable strategies found"
        raise ValueError(msg)

    # Phase 1: coarse grid
    log(f"Phase 1: coarse search over {', '.join(names)}...")
    coarse_grids = {
        name: _build_grid(PARAM_RANGES[name], "coarse") for name in names
    }

    # Total combinations = product of each strategy's grid size
    shared = (names, portfolio, price_data, start, end, log)

    all_combos = list(
        itertools.product(*(coarse_grids[n] for n in names))
    )
    best_return, best_params, best_bh, best_train, best_test = _run_parallel(
        all_combos, *shared
    )
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
    ret2, params2, bh2, train2, test2 = _run_parallel(fine_combos, *shared)
    if ret2 > best_return:
        best_return, best_params, best_bh, best_train, best_test = (
            ret2, params2, bh2, train2, test2
        )
    log(f"  Phase 2 best: {best_return:.2%}")

    trials_total = len(all_combos) + len(fine_combos)

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
        clean = {}
        for k, v in p.items():
            clean[k] = int(v) if k in _INT_PARAMS else round(v, 4)
        strategies.append({"name": name, "params": clean})

    with open(path, "w") as f:
        yaml.dump({"strategies": strategies}, f, default_flow_style=False)
