"""Bayesian optimizer for strategy parameters using Optuna TPE."""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date

import optuna
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

# Meta-params prefixed with _ are not passed to the strategy constructor.
# _weight: blending weight for conviction strategies.
# _veto_threshold: veto threshold for protective strategies.
_META_PARAMS = {"_weight", "_veto_threshold"}

# Synthetic key for global allocation knobs (sigmoid_steepness, rebalance_threshold).
_GLOBAL_KEY = "_global"

# Default parameter ranges per strategy.
# Each entry: (min, max, step) — step used for discretisation.
PARAM_RANGES: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "MeanReversion": {
        "window": (10, 100, 10, 5),
        "threshold": (0.03, 0.25, 0.03, 0.01),
        "_weight": (0.5, 3.0, 0.5, 0.25),
    },
    "ProfitTaking": {
        "gain_threshold": (0.10, 0.80, 0.10, 0.03),
        "_weight": (0.5, 3.0, 0.5, 0.25),
    },
    "Momentum": {
        "window": (5, 49, 5, 2),
        "_weight": (0.5, 3.0, 0.5, 0.25),
    },
    "RSIOversold": {
        "window": (7, 28, 7, 2),
        "oversold_threshold": (15.0, 40.0, 5.0, 2.0),
        "_weight": (0.5, 3.0, 0.5, 0.25),
    },
    "RSIOverbought": {
        "window": (7, 28, 7, 2),
        "overbought_threshold": (60.0, 85.0, 5.0, 2.0),
        "_weight": (0.5, 3.0, 0.5, 0.25),
    },
    "BollingerBand": {
        "window": (10, 50, 10, 5),
        "num_std": (1.5, 3.0, 0.5, 0.25),
        "_weight": (0.5, 3.0, 0.5, 0.25),
    },
    "MACDCrossover": {
        "fast_period": (8, 16, 4, 2),
        "slow_period": (20, 32, 4, 2),
        "signal_period": (5, 13, 4, 2),
        "_weight": (0.5, 3.0, 0.5, 0.25),
    },
    "DollarCostAveraging": {
        "frequency_days": (5, 30, 5, 2),
    },
    "GapDownRecovery": {
        "gap_threshold": (0.02, 0.08, 0.02, 0.005),
        "_weight": (0.5, 3.0, 0.5, 0.25),
    },
    "TrailingStop": {
        "trail_pct": (0.05, 0.25, 0.05, 0.02),
        "_veto_threshold": (-0.8, -0.2, 0.2, 0.1),
    },
    "StopLoss": {
        "loss_threshold": (0.05, 0.25, 0.05, 0.02),
        "_veto_threshold": (-0.8, -0.2, 0.2, 0.1),
    },
    "VWAPReversion": {
        "window": (10, 50, 10, 5),
        "threshold": (0.01, 0.05, 0.01, 0.005),
        "_weight": (0.5, 3.0, 0.5, 0.25),
    },
    "MovingAverageCrossover": {
        "short_window": (10, 30, 5, 2),
        "long_window": (40, 100, 10, 5),
        "_weight": (0.5, 3.0, 0.5, 0.25),
    },
    _GLOBAL_KEY: {
        "sigmoid_steepness": (1.0, 5.0, 1.0, 0.5),
        "rebalance_threshold": (0.01, 0.05, 0.01, 0.005),
    },
}

_DEFAULT_N_TRIALS = 200


@dataclass
class OptimizeResult:
    best_params: dict[str, dict[str, float]]
    best_return: float
    best_bh_return: float
    best_train_return: float
    best_test_return: float
    trials_run: int


def _suggest_params(
    trial: optuna.Trial,
    strategy_name: str,
    ranges: dict[str, tuple[float, float, float, float]],
) -> dict[str, float]:
    """Use Optuna trial to suggest parameter values for one strategy."""
    params: dict[str, float] = {}
    for param, (lo, hi, _coarse_step, fine_step) in ranges.items():
        key = f"{strategy_name}__{param}"
        if param in _INT_PARAMS:
            params[param] = float(
                trial.suggest_int(key, int(lo), int(hi), step=int(fine_step))
            )
        else:
            params[param] = trial.suggest_float(key, lo, hi, step=fine_step)
    return params


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
    # Extract global allocation knobs
    global_params = strategy_params.get(_GLOBAL_KEY, {})
    conviction: list[tuple] = []
    protective: list[tuple] = []
    mechanical = []

    for name, params in strategy_params.items():
        if name == _GLOBAL_KEY:
            continue
        cls = STRATEGY_REGISTRY[name]
        # Separate meta-params from constructor params
        weight = params.get("_weight", 1.0)
        veto_threshold = params.get("_veto_threshold", -0.5)
        clean_params = {
            k: int(v) if k in _INT_PARAMS else v
            for k, v in params.items()
            if k not in _META_PARAMS
        }
        strategy = cls(**clean_params)

        if strategy.tier == StrategyTier.PROTECTIVE:
            protective.append((strategy, veto_threshold))
        elif strategy.tier == StrategyTier.MECHANICAL:
            mechanical.append(strategy)
        else:
            conviction.append((strategy, weight))

    # Count tickers in portfolio
    n_tickers = sum(1 for h in portfolio.holdings if h.shares > 0)
    constraints = AllocationConstraints(
        sigmoid_steepness=global_params.get("sigmoid_steepness", 2.0),
        rebalance_threshold=global_params.get("rebalance_threshold", 0.02),
    )

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


def _init_worker(
    portfolio: PortfolioConfig,
    price_data: dict[str, pd.Series],
    start: date,
    end: date,
) -> None:
    _worker_state.update(
        portfolio=portfolio, price_data=price_data, start=start, end=end
    )


def _trial_worker(strategy_params: dict[str, dict[str, float]]) -> tuple[float, ...]:
    return _run_trial(strategy_params, **_worker_state)


def optimize(
    portfolio: PortfolioConfig,
    price_data: dict[str, pd.Series],
    start: date,
    end: date,
    strategy_names: list[str] | None = None,
    n_trials: int = _DEFAULT_N_TRIALS,
    log_fn: Callable[[str], None] | None = None,
) -> OptimizeResult:
    """Bayesian optimization over strategy parameters using Optuna TPE.

    Runs *n_trials* Optuna trials (default 200).  Each trial samples a
    parameter combination via the Tree-structured Parzen Estimator and
    evaluates it with a full backtest.  Backtests are executed in a worker
    pool to utilise multiple CPU cores.
    """
    log = log_fn or (lambda _: None)

    names = strategy_names or [
        k for k in PARAM_RANGES if k != _GLOBAL_KEY
    ]
    # Filter to strategies that have defined ranges and are not MECHANICAL
    names = [
        n for n in names
        if n in PARAM_RANGES
        and STRATEGY_REGISTRY[n]().tier != StrategyTier.MECHANICAL
    ]

    if not names:
        msg = "No optimizable strategies found"
        raise ValueError(msg)

    # Always include global allocation knobs
    names.append(_GLOBAL_KEY)

    max_workers = min((os.cpu_count() or 4) // 2, n_trials) or 1

    log(f"Optimizing {', '.join(names)} — {n_trials} trials ({max_workers} workers)")

    # Suppress Optuna's default logging (we provide our own via log_fn).
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # -- Objective that runs in the main process but farms backtest to pool --
    pool = ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(portfolio, price_data, start, end),
    )

    trials_done = 0

    def objective(trial: optuna.Trial) -> float:
        nonlocal trials_done

        strategy_params: dict[str, dict[str, float]] = {}
        for name in names:
            strategy_params[name] = _suggest_params(
                trial, name, PARAM_RANGES[name],
            )

        total_ret, bh_ret, train_ret, test_ret = pool.submit(
            _trial_worker, strategy_params,
        ).result()

        # Store auxiliary metrics as user attributes for later retrieval.
        trial.set_user_attr("bh_return", bh_ret)
        trial.set_user_attr("train_return", train_ret)
        trial.set_user_attr("test_return", test_ret)
        trial.set_user_attr("params", strategy_params)

        trials_done += 1
        if trials_done % 25 == 0 or trials_done == n_trials:
            log(f"  {trials_done}/{n_trials} — best so far: {study.best_value:.2%}")

        return total_ret

    try:
        study.optimize(objective, n_trials=n_trials)
    finally:
        pool.shutdown(wait=False)

    best = study.best_trial
    best_params: dict[str, dict[str, float]] = best.user_attrs["params"]

    log(f"Best return: {best.value:.2%} in {len(study.trials)} trials")

    return OptimizeResult(
        best_params=best_params,
        best_return=round(best.value, 4),
        best_bh_return=round(best.user_attrs["bh_return"], 4),
        best_train_return=round(best.user_attrs["train_return"], 4),
        best_test_return=round(best.user_attrs["test_return"], 4),
        trials_run=len(study.trials),
    )


def write_strategies_yaml(
    params: dict[str, dict[str, float]], path: str
) -> None:
    """Write optimized parameters to a strategies YAML file."""
    output: dict[str, object] = {}

    # Emit global allocation knobs as top-level keys
    if _GLOBAL_KEY in params:
        for k, v in params[_GLOBAL_KEY].items():
            output[k] = round(v, 4)

    strategies = []
    for name, p in params.items():
        if name == _GLOBAL_KEY:
            continue
        entry: dict[str, object] = {"name": name}
        clean_params: dict[str, object] = {}
        for k, v in p.items():
            if k == "_weight":
                entry["weight"] = round(v, 4)
            elif k == "_veto_threshold":
                entry["veto_threshold"] = round(v, 4)
            elif k in _INT_PARAMS:
                clean_params[k] = int(v)
            else:
                clean_params[k] = round(v, 4)
        if clean_params:
            entry["params"] = clean_params
        strategies.append(entry)

    output["strategies"] = strategies

    with open(path, "w") as f:
        yaml.dump(output, f, default_flow_style=False)
