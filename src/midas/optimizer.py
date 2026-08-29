"""Bayesian optimizer for strategy parameters using Optuna TPE."""

from __future__ import annotations

import decimal
import logging
import math
import os
import statistics
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date
from typing import Any, get_args

import optuna
import pandas as pd
import yaml

from midas.allocator import Allocator
from midas.backtest import DEFAULT_TRAIN_PCT, BacktestEngine
from midas.metrics import DAYS_PER_YEAR, compute_annualized_return
from midas.models import (
    DEFAULT_MIN_CASH_PCT,
    DEFAULT_OBJECTIVE,
    AllocationConstraints,
    ForecastScaling,
    Objective,
    PortfolioConfig,
    RiskConfig,
    TaxConfig,
    objective_error,
)
from midas.order_sizer import OrderSizer
from midas.results import BacktestResult
from midas.strategies import STRATEGY_REGISTRY
from midas.strategies.base import EntrySignal, ExitRule

# Parameters that should be cast to int when building strategy instances.
INT_PARAMS = {
    "window",
    "short_window",
    "long_window",
    "fast_period",
    "slow_period",
    "signal_period",
    "frequency_days",
}

# Meta-params prefixed with _ are not passed to the strategy constructor.
# _weight: blending weight for entry signals.
META_PARAMS = {"_weight"}

# Synthetic key for global allocation knobs (softmax_temperature, min_buy_delta).
ALLOCATION_KEY = "_global"

# Default parameter ranges per strategy.
# Each entry: (min, max, step) — step used for Optuna discretisation.
PARAM_RANGES: dict[str, dict[str, tuple[float, float, float]]] = {
    # --- Entry signals (use _weight for blending) ---
    "MeanReversion": {
        "window": (10, 100, 5),
        "threshold": (0.03, 0.25, 0.01),
        "_weight": (0.5, 3.0, 0.25),
    },
    "Momentum": {
        "window": (5, 50, 2),
        "momentum_scale": (0.02, 0.10, 0.01),
        "_weight": (0.5, 3.0, 0.25),
    },
    "RSIOversold": {
        "window": (7, 28, 2),
        "oversold_threshold": (15.0, 40.0, 2.0),
        "_weight": (0.5, 3.0, 0.25),
    },
    "BollingerBand": {
        "window": (10, 50, 5),
        "num_std": (1.5, 3.0, 0.25),
        "_weight": (0.5, 3.0, 0.25),
    },
    "DonchianBreakout": {
        "window": (10, 60, 5),
        "breakout_scale": (0.01, 0.06, 0.005),
        "_weight": (0.5, 3.0, 0.25),
    },
    "MACDCrossover": {
        "fast_period": (8, 16, 2),
        "slow_period": (20, 40, 2),
        "signal_period": (5, 13, 2),
        "_weight": (0.5, 3.0, 0.25),
    },
    "GapDownRecovery": {
        "gap_threshold": (0.02, 0.08, 0.005),
        "_weight": (0.5, 3.0, 0.25),
    },
    "KeltnerChannel": {
        "window": (10, 50, 5),
        "multiplier": (1.0, 4.0, 0.25),
        "_weight": (0.5, 3.0, 0.25),
    },
    "VWAPReversion": {
        "window": (10, 50, 5),
        "threshold": (0.01, 0.05, 0.005),
        "_weight": (0.5, 3.0, 0.25),
    },
    "MovingAverageCrossover": {
        "short_window": (10, 30, 2),
        "long_window": (40, 100, 5),
        "spread_scale": (0.02, 0.10, 0.01),
        "_weight": (0.5, 3.0, 0.25),
    },
    # --- Exit rules (no _weight — exits don't participate in blending) ---
    "ProfitTaking": {
        "gain_threshold": (0.10, 0.80, 0.03),
    },
    "TrailingStop": {
        "trail_pct": (0.05, 0.25, 0.02),
    },
    "StopLoss": {
        "loss_threshold": (0.05, 0.25, 0.02),
    },
    "ChandelierStop": {
        "window": (10, 40, 2),
        "multiplier": (1.5, 5.0, 0.25),
    },
    "MACDExit": {
        "fast_period": (8, 16, 2),
        "slow_period": (20, 40, 2),
        "signal_period": (5, 13, 2),
    },
    "MovingAverageCrossoverExit": {
        "short_window": (10, 30, 2),
        "long_window": (40, 100, 5),
    },
    "ParabolicSARExit": {
        "af_start": (0.01, 0.04, 0.005),
        "af_step": (0.01, 0.04, 0.005),
        "af_max": (0.10, 0.30, 0.02),
    },
    ALLOCATION_KEY: {
        # softmax_temperature: low = concentrated, high = uniform split.
        "softmax_temperature": (0.2, 1.0, 0.1),
        "min_buy_delta": (0.01, 0.05, 0.005),
        # min_cash_pct is a user risk preference, not optimized
        # max_position_pct is computed dynamically in optimize() from n_tickers
    },
}

DEFAULT_N_TRIALS = 200


def entry_search_names() -> list[str]:
    """Entry-signal strategies the optimizer may search.

    Exit rules and allocator globals are policy-owned: the operator sets
    them and the search never samples them, so ensembles differ only in
    forecasts and the engine has exactly one sizing/exit configuration.
    """
    return [
        name for name in PARAM_RANGES if name != ALLOCATION_KEY and issubclass(STRATEGY_REGISTRY[name], EntrySignal)
    ]


# Sampler seed: fixed so a given (inputs, seed) pair reproduces its search
# byte for byte. Walk-forward offsets it per fold so folds stay distinct.
DEFAULT_SEED = 42

# Below this many trials per fold, TPE has not converged over the parameter
# space and per-fold results are dominated by sampler noise rather than by
# the parameters. Measured on a 10-year walk-forward: 30 trials/fold gave a
# 7.6-point spread in OOS CAGR across sampler seeds, 100 gave 2.9 — while
# the median barely moved. Differences smaller than that spread (objective
# choice, for one) are unmeasurable below it.
MIN_TRIALS_PER_FOLD = 100

# Walk-forward defaults: reserve 60% for the first training window,
# then carve the remaining 40% into test windows of ~63 trading days
# (~3 calendar months) each.
WF_MIN_TRAIN_PCT = 0.60
WF_MIN_TEST_DAYS = 63


CALMAR_DRAWDOWN_FLOOR = 0.01
ULCER_FLOOR = 0.005


@dataclass(frozen=True)
class TrialMetrics:
    """In-sample metrics from one trial backtest, scored by :func:`score_trial`.

    Every field is measured over the training window only — the worker runs
    the trial on ``[start, train_end]`` with no internal split, so nothing
    here has seen the test period. ``after_tax_twr`` is ``None`` when the
    portfolio has no tax config.
    """

    twr: float
    bh_return: float
    sharpe_ratio: float
    after_tax_twr: float | None
    max_drawdown: float = 0.0
    ulcer_index: float = 0.0
    block_returns: tuple[float, ...] = ()


def score_trial(metrics: TrialMetrics, objective: Objective) -> float:
    """Pick the scalar Optuna maximizes for *objective*.

    Args:
        metrics: In-sample metrics from one trial backtest.
        objective: ``gross`` (raw TWR), ``sharpe`` (annualized Sharpe),
            ``net`` (TWR after tax), ``calmar`` (TWR over max drawdown),
            ``ulcer`` (TWR over Ulcer Index), or ``robust`` (lower-quartile
            block return).

    Returns:
        The value to maximize; higher is better for every objective.

    Raises:
        ValueError: On an unknown objective, or ``net`` without tax accounting.
    """
    if objective == "gross":
        return metrics.twr
    if objective == "sharpe":
        return metrics.sharpe_ratio
    if objective == "net":
        if metrics.after_tax_twr is None:
            msg = "objective 'net' requires a tax: block in the portfolio YAML (after-tax accounting is off)"
            raise ValueError(msg)
        return metrics.after_tax_twr
    if objective == "calmar":
        # Floor the drawdown: a monotonic train window has ~zero drawdown and
        # an unfloored ratio would be inf, dominating every real candidate.
        # As with any return-over-risk ratio, losers rank inversely (a -5%
        # return looks better with a 20% drawdown than with a 5% one); the
        # best trial is always a winner, so only the model of the losing
        # region is distorted.
        return metrics.twr / max(metrics.max_drawdown, CALMAR_DRAWDOWN_FLOOR)
    if objective == "ulcer":
        # Martin ratio; same floor rationale as calmar.
        return metrics.twr / max(metrics.ulcer_index, ULCER_FLOOR)
    if objective == "robust":
        blocks = metrics.block_returns
        if len(blocks) >= 2:
            # Inclusive so the quartile stays within [min, max] even with two
            # or three blocks; the exclusive default extrapolates below the
            # worst block there.
            return statistics.quantiles(blocks, n=4, method="inclusive")[0]
        return blocks[0] if blocks else 0.0
    raise ValueError(objective_error(objective))


@dataclass
class OptimizeResult:
    """Best parameters and headline figures from a single optimisation run.

    ``best_objective_value`` is whatever *objective* maximized (a return for
    ``gross``/``net``, a ratio for ``sharpe``); ``best_train_return`` is always
    the in-sample TWR so the train/test comparison stays return-based.
    """

    best_params: dict[str, dict[str, float]]
    objective: Objective
    best_objective_value: float
    best_bh_return: float
    best_train_return: float
    best_test_return: float
    trials_run: int
    best_result: BacktestResult | None = None


@dataclass
class FoldResult:
    """Result of a single walk-forward fold.

    ``train_return`` and ``test_return`` are annualized (so folds of different
    lengths compare apples-to-apples). ``train_return_raw`` and
    ``test_return_raw`` are the raw (un-annualized) TWR over the fold's own
    window — used by the overall-CAGR compounding loop, which must not
    double-annualize.

    The ``after_tax_*`` pair mirrors the test-return pair when the portfolio
    has a tax config, else ``None``. Each fold is a fresh backtest, so lots
    re-enter at the fold start and every gain inside a short fold is
    short-term: per-fold after-tax figures overstate tax drag relative to
    a continuous run, but compare objectives on equal footing.
    """

    fold: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    best_params: dict[str, dict[str, float]]
    train_return: float
    test_return: float
    train_return_raw: float
    test_return_raw: float
    trials_run: int
    objective_value: float = 0.0  # what the fold's search maximized (train window)
    after_tax_test_return: float | None = None
    after_tax_test_return_raw: float | None = None
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0


@dataclass
class WalkForwardResult:
    """Aggregated result of walk-forward optimisation across all folds."""

    folds: list[FoldResult]
    annualized_return: float  # CAGR over the OOS period
    mean_test_return: float
    std_test_return: float
    winning_folds: int  # number of folds with positive OOS return
    best_fold_return: float  # best single-fold OOS return
    worst_fold_return: float  # worst single-fold OOS return
    efficiency_ratio: float  # mean OOS return / mean train return
    best_params: dict[str, dict[str, float]]  # from last fold (most recent data)
    total_trials: int
    objective: Objective = DEFAULT_OBJECTIVE
    after_tax_annualized_return: float | None = None  # OOS CAGR after tax; None without a tax config
    mean_max_drawdown: float = 0.0
    mean_sharpe: float = 0.0
    mean_sortino: float = 0.0
    mean_win_rate: float = 0.0


def _snap_high(lo: float, hi: float, step: float) -> float:
    """Snap ``hi`` down so ``(hi - lo)`` is divisible by ``step``.

    Uses Decimal to match Optuna's own divisibility check and avoid float noise.
    """
    d_lo, d_hi, d_step = decimal.Decimal(str(lo)), decimal.Decimal(str(hi)), decimal.Decimal(str(step))
    return float((d_hi - d_lo) // d_step * d_step + d_lo)


def _pool_workers(n_trials: int) -> int:
    """Worker-pool size: half the CPUs, capped by the trial count, at least one."""
    return min((os.cpu_count() or 4) // 2, n_trials) or 1


def _suggest_params(
    trial: optuna.Trial,
    strategy_name: str,
    ranges: dict[str, tuple[float, float, float]],
) -> dict[str, float]:
    """Use Optuna trial to suggest parameter values for one strategy."""
    params: dict[str, float] = {}
    for param, (lo, hi, step) in ranges.items():
        key = f"{strategy_name}__{param}"
        # Snap hi down so (hi - lo) is divisible by step, avoiding Optuna warnings.
        hi = _snap_high(lo, hi, step)
        if param in INT_PARAMS:
            params[param] = float(trial.suggest_int(key, int(lo), int(hi), step=int(step)))
        else:
            params[param] = trial.suggest_float(key, lo, hi, step=step)
    return params


def _instantiate_strategies(
    strategy_params: dict[str, dict[str, float]],
) -> tuple[list[tuple[EntrySignal, float]], list[ExitRule]]:
    """Build entry and exit strategy instances from sampled parameter dicts.

    Raises:
        TypeError: If a named strategy is neither an EntrySignal nor an ExitRule.
    """
    entries: list[tuple[EntrySignal, float]] = []
    exits: list[ExitRule] = []

    for name, params in strategy_params.items():
        if name == ALLOCATION_KEY:
            continue
        cls = STRATEGY_REGISTRY[name]
        weight = params.get("_weight", 1.0)
        clean_params = {
            key: int(val) if key in INT_PARAMS else val for key, val in params.items() if key not in META_PARAMS
        }
        strategy = cls(**clean_params)

        if isinstance(strategy, ExitRule):
            exits.append(strategy)
        elif isinstance(strategy, EntrySignal):
            entries.append((strategy, weight))
        else:
            msg = f"Strategy {name!r} is neither EntrySignal nor ExitRule"
            raise TypeError(msg)

    return entries, exits


def _run_trial(
    strategy_params: dict[str, dict[str, float]],
    portfolio: PortfolioConfig,
    price_data: dict[str, pd.DataFrame],
    start: date,
    end: date,
    min_cash_pct: float = DEFAULT_MIN_CASH_PCT,
    train_pct: float = DEFAULT_TRAIN_PCT,
    enable_split: bool = True,
    risk_config: RiskConfig | None = None,
    forecast_scaling: ForecastScaling = "none",
    tax_config: TaxConfig | None = None,
    track_vol_contribution: bool = True,
) -> tuple[TrialMetrics, BacktestResult]:
    """Run a single backtest trial with the allocator + order_sizer + exit_rules system.

    Returns the scoring metrics alongside the full result. Callers that want
    in-sample-only metrics must pass a train-only window with
    ``enable_split=False``; the metrics cover whatever window was run.
    """
    # Extract global allocation knobs
    global_params = strategy_params.get(ALLOCATION_KEY, {})
    entries, exits = _instantiate_strategies(strategy_params)

    constraints = AllocationConstraints(
        max_position_pct=global_params.get("max_position_pct"),
        min_cash_pct=min_cash_pct,
        softmax_temperature=global_params.get("softmax_temperature", 0.5),
        min_buy_delta=global_params.get("min_buy_delta", 0.02),
        forecast_scaling=forecast_scaling,
    )

    allocator = Allocator(entries, constraints, portfolio.active_ticker_count(), risk_config=risk_config)
    order_sizer = OrderSizer()

    engine = BacktestEngine(
        allocator=allocator,
        order_sizer=order_sizer,
        exit_rules=exits,
        constraints=constraints,
        train_pct=train_pct,
        enable_split=enable_split,
        tax_config=tax_config,
        track_vol_contribution=track_vol_contribution,
    )
    result = engine.run(portfolio, price_data, start, end)

    # ``after_tax_twr`` is None for two unrelated reasons: accounting is off
    # (no tax config) or the run was degenerate (nothing to invest, empty
    # curve). Only the first is a configuration error ``score_trial`` should
    # raise on; a degenerate trial scores 0.0 under ``net`` exactly as it
    # does under ``gross``/``sharpe`` instead of aborting the whole study.
    after_tax_twr: float | None = None
    if tax_config is not None:
        after_tax_twr = result.after_tax_twr if result.after_tax_twr is not None else 0.0

    if result.starting_value <= 0:
        return TrialMetrics(twr=0.0, bh_return=0.0, sharpe_ratio=0.0, after_tax_twr=after_tax_twr), result

    bh_return = (result.buy_and_hold_value - result.starting_value) / result.starting_value
    metrics = TrialMetrics(
        twr=result.twr if result.twr is not None else 0.0,
        bh_return=bh_return,
        sharpe_ratio=result.sharpe_ratio,
        after_tax_twr=after_tax_twr,
        max_drawdown=result.max_drawdown,
        ulcer_index=result.ulcer_index,
        block_returns=tuple(result.block_returns),
    )
    return metrics, result


def _trading_days(price_data: dict[str, pd.DataFrame], start: date, end: date) -> list[date]:
    """Sorted union of trading days across all tickers within ``[start, end]``."""
    all_dates: set[date] = set()
    for df in price_data.values():
        all_dates.update(dt for dt in df.index if start <= dt <= end)
    return sorted(all_dates)


def train_window_end(trading_days: list[date], train_pct: float) -> date:
    """Last trading day of the training window, mirroring the engine's split.

    The engine places the split at ``trading_days[int(n * train_pct)]`` and
    counts days strictly before it as training, so a trial run on
    ``[start, train_window_end]`` with no internal split sees at most the
    in-sample days of a full-range split run — and nothing from the test
    side. (The split run snapshots its train value at the split day's
    close, one bar later than this window ends, so the reported
    ``train_return`` can differ from the searched value by that bar.)

    Args:
        trading_days: Sorted union of trading days over the full range.
        train_pct: Fraction of days reserved for training, in ``(0, 1]``.

    Returns:
        The last training day. ``train_pct=1.0`` means the whole range.

    Raises:
        ValueError: On no trading days, or a ``train_pct`` so small that the
            split leaves no training days — without this guard the index
            would wrap to the last day and score every trial on the full range.
    """
    if not trading_days:
        msg = "No trading days in the requested range"
        raise ValueError(msg)
    split_idx = int(len(trading_days) * train_pct)
    if split_idx == 0:
        msg = f"train_pct {train_pct} leaves no training days in {len(trading_days)} trading days"
        raise ValueError(msg)
    if split_idx >= len(trading_days):
        return trading_days[-1]
    return trading_days[split_idx - 1]


def format_objective(value: float, objective: Objective) -> str:
    """Render an objective value for logs: ratios as decimals, returns as percents."""
    return f"{value:.2%}" if objective in ("gross", "net", "robust") else f"{value:.2f}"


def _require_tax_for_net(objective: Objective, tax_config: TaxConfig | None) -> None:
    """Fail before any work starts when ``net`` has nothing to tax.

    Raises:
        ValueError: On an unknown objective, or ``net`` without a tax config.
    """
    if objective not in get_args(Objective.__value__):
        raise ValueError(objective_error(objective))
    if objective == "net" and tax_config is None:
        msg = "objective 'net' requires a tax: block in the portfolio YAML (after-tax accounting is off)"
        raise ValueError(msg)


worker_state: dict[str, Any] = {}


def _init_worker(
    portfolio: PortfolioConfig,
    price_data: dict[str, pd.DataFrame],
    start: date,
    train_end: date,
    min_cash_pct: float,
    risk_config: RiskConfig | None = None,
    forecast_scaling: ForecastScaling = "none",
    tax_config: TaxConfig | None = None,
) -> None:
    """Initialise standard-optimizer workers with the train-window trial state."""
    # Suppress allocator warnings during trial evaluation — the optimizer
    # explores boundary values that trigger heuristic warnings but are fine
    # to evaluate. Scoped to the worker process so the main-process final
    # re-run at the end of optimize() still surfaces warnings for the chosen
    # config, and subsequent backtest/live calls in the same session aren't
    # silently muted.
    logging.getLogger("midas.allocator").setLevel(logging.ERROR)
    worker_state.update(
        portfolio=portfolio,
        price_data=price_data,
        start=start,
        end=train_end,
        min_cash_pct=min_cash_pct,
        enable_split=False,
        risk_config=risk_config,
        forecast_scaling=forecast_scaling,
        tax_config=tax_config,
    )


def _trial_worker(strategy_params: dict[str, dict[str, float]]) -> TrialMetrics:
    """Evaluate one trial in a worker process, dropping the unpicklable result.

    Vol attribution feeds only the report table on the final re-run; trials
    skip it (it is the single hottest per-bar cost in a backtest).
    """
    metrics, _result = _run_trial(strategy_params, track_vol_contribution=False, **worker_state)
    return metrics


def _wf_init_worker(
    portfolio: PortfolioConfig,
    price_data: dict[str, pd.DataFrame],
    min_cash_pct: float,
    risk_config: RiskConfig | None = None,
    forecast_scaling: ForecastScaling = "none",
    tax_config: TaxConfig | None = None,
) -> None:
    """Initialise walk-forward workers with static state only (dates vary per call)."""
    logging.getLogger("midas.allocator").setLevel(logging.ERROR)
    worker_state.update(
        portfolio=portfolio,
        price_data=price_data,
        min_cash_pct=min_cash_pct,
        risk_config=risk_config,
        forecast_scaling=forecast_scaling,
        tax_config=tax_config,
    )


def _wf_trial_worker(
    strategy_params: dict[str, dict[str, float]],
    start: date,
    end: date,
) -> TrialMetrics:
    """Evaluate one walk-forward trial over the given window in a worker process."""
    metrics, _result = _run_trial(
        strategy_params,
        worker_state["portfolio"],
        worker_state["price_data"],
        start,
        end,
        worker_state["min_cash_pct"],
        enable_split=False,
        risk_config=worker_state.get("risk_config"),
        forecast_scaling=worker_state.get("forecast_scaling", "none"),
        tax_config=worker_state.get("tax_config"),
        track_vol_contribution=False,
    )
    return metrics


def max_warmup_for_search(
    strategy_names: list[str] | None,
    min_cash_pct: float = DEFAULT_MIN_CASH_PCT,
    n_tickers: int = 1,
) -> int:
    """Upper bound on warmup bars across the optimizer's search space.

    Walk-forward and standard optimizer trials sample different parameter
    values, so the prefetched warmup buffer must cover the worst case.
    For each optimizable strategy, instantiate it with every integer
    parameter pinned to its search-range upper bound and take the max
    ``warmup_period``.
    """
    names, ranges = _prepare_names_and_ranges(strategy_names, min_cash_pct, n_tickers)
    max_warmup_val = 0
    for name in names:
        if name == ALLOCATION_KEY:
            continue
        cls = STRATEGY_REGISTRY[name]
        params: dict[str, Any] = {}
        for pname, (lo, hi, step) in ranges.get(name, {}).items():
            if pname in META_PARAMS:
                continue
            # Snap hi down the same way _suggest_params does so the upper
            # bound here matches values the optimizer actually tries.
            snapped_hi = _snap_high(lo, hi, step)
            params[pname] = int(snapped_hi) if pname in INT_PARAMS else snapped_hi
        # Let TypeError propagate — if a search-range key doesn't match a
        # constructor param, that's a real configuration bug we want loud.
        instance = cls(**params)
        max_warmup_val = max(max_warmup_val, instance.warmup_period)
    return max_warmup_val


def _prepare_names_and_ranges(
    strategy_names: list[str] | None,
    min_cash_pct: float,
    n_tickers: int,
    search_globals: bool = False,
) -> tuple[list[str], dict[str, dict[str, tuple[float, float, float]]]]:
    """Resolve searched strategies and their ranges (shared by optimize/walk-forward).

    By default only entry signals are searched; exit rules and allocator
    globals are policy-owned. ``search_globals=True`` restores the full
    legacy space for experiments.

    Raises:
        ValueError: If no searchable strategies remain, or a policy-owned
            name is requested without ``search_globals``.
    """
    searchable = set(PARAM_RANGES) if search_globals else set(entry_search_names())
    names = strategy_names or sorted(searchable - {ALLOCATION_KEY})
    rejected = [n for n in names if n in PARAM_RANGES and n not in searchable]
    if rejected:
        msg = f"{sorted(rejected)} are policy-owned (exit rules); set search_globals to search them"
        raise ValueError(msg)
    names = [name for name in names if name in searchable]

    if not names:
        msg = "No optimizable strategies found"
        raise ValueError(msg)

    ranges = {name: dict(PARAM_RANGES[name]) for name in names}

    if search_globals:
        names.append(ALLOCATION_KEY)
        equal_weight = (1.0 - min_cash_pct) / max(n_tickers, 1)
        lo = max(round(1.5 * equal_weight, 2), 0.10)
        hi = min(round(5.0 * equal_weight, 2), 0.80)
        if lo >= hi:
            lo, hi = 0.10, 0.80
        step = round((hi - lo) / 8, 2) or 0.01
        ranges.setdefault(ALLOCATION_KEY, dict(PARAM_RANGES[ALLOCATION_KEY]))
        ranges[ALLOCATION_KEY]["max_position_pct"] = (lo, hi, step)

    return names, ranges


def _run_trials_batched(
    study: optuna.Study,
    names: list[str],
    ranges: dict[str, dict[str, tuple[float, float, float]]],
    n_trials: int,
    batch_size: int,
    objective: Objective,
    submit: Callable[[dict[str, dict[str, float]]], Future[TrialMetrics]],
    log: Callable[[str], None],
) -> None:
    """Run *n_trials* through *study* with deterministic ask/tell batching.

    ``study.optimize(n_jobs=k)`` races k threads, so the sampler sees a
    different trial-history order every run and a fixed seed reproduces
    nothing. Here trials are asked sequentially in the main thread
    (deterministic sampling), evaluated concurrently in the worker pool,
    and told back in ask order — same parallelism, reproducible search.
    """
    done = 0
    while done < n_trials:
        batch: list[tuple[optuna.Trial, dict[str, dict[str, float]], Future[TrialMetrics]]] = []
        for _ in range(min(batch_size, n_trials - done)):
            trial = study.ask()
            params = {name: _suggest_params(trial, name, ranges[name]) for name in names}
            batch.append((trial, params, submit(params)))
        for trial, params, future in batch:
            metrics = future.result()
            trial.set_user_attr("bh_return", metrics.bh_return)
            trial.set_user_attr("train_return", metrics.twr)
            trial.set_user_attr("params", params)
            study.tell(trial, score_trial(metrics, objective))
            done += 1
            if done % 25 == 0 or done == n_trials:
                pct = done * 100 // n_trials
                best = format_objective(study.best_value, objective)
                log(f"  [{pct:3d}%] {done}/{n_trials} trials — best {objective}: {best}")


def optimize(
    portfolio: PortfolioConfig,
    price_data: dict[str, pd.DataFrame],
    start: date,
    end: date,
    strategy_names: list[str] | None = None,
    n_trials: int = DEFAULT_N_TRIALS,
    min_cash_pct: float = DEFAULT_MIN_CASH_PCT,
    train_pct: float = DEFAULT_TRAIN_PCT,
    log_fn: Callable[[str], None] | None = None,
    risk_config: RiskConfig | None = None,
    forecast_scaling: ForecastScaling = "none",
    objective: Objective = DEFAULT_OBJECTIVE,
    tax_config: TaxConfig | None = None,
    seed: int = DEFAULT_SEED,
) -> OptimizeResult:
    """Bayesian optimization over strategy parameters using Optuna TPE.

    Runs *n_trials* Optuna trials (default 200).  Each trial samples a
    parameter combination via the Tree-structured Parzen Estimator and
    evaluates it with a backtest over the training window only — the first
    *train_pct* of trading days — so the objective never sees the test
    period.  Backtests are executed in a worker pool to utilise multiple
    CPU cores.  The best parameters are then re-run over the full range with
    the train/test split for reporting.

    Raises:
        ValueError: On an unknown objective, or ``net`` without a tax config.
    """
    log = log_fn or (lambda _: None)
    _require_tax_for_net(objective, tax_config)

    names, ranges = _prepare_names_and_ranges(strategy_names, min_cash_pct, portfolio.active_ticker_count())
    train_end = train_window_end(_trading_days(price_data, start, end), train_pct)

    max_workers = _pool_workers(n_trials)

    strat_names = [name for name in names if name != ALLOCATION_KEY]
    log(f"Optimizing {len(strat_names)} strategies over {start} to {end}")
    log(f"  objective: {objective} | train window: {start} → {train_end}")
    log(f"  {n_trials} trials across {max_workers} workers")

    # Suppress Optuna's default logging (we provide our own via log_fn).
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    # -- Objective that runs in the main process but farms backtest to pool --
    pool = ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(portfolio, price_data, start, train_end, min_cash_pct, risk_config, forecast_scaling, tax_config),
    )

    try:
        _run_trials_batched(
            study,
            names,
            ranges,
            n_trials,
            batch_size=max_workers,
            objective=objective,
            submit=lambda params: pool.submit(_trial_worker, params),
            log=log,
        )
    finally:
        pool.shutdown(wait=True)

    best = study.best_trial
    best_params: dict[str, dict[str, float]] = best.user_attrs["params"]

    best_value = best.value or 0.0
    log(f"Optimization complete — best {objective}: {format_objective(best_value, objective)}")

    # Re-run best params over the full range with the split in the main
    # process: the report wants train/test on one continuous equity curve,
    # and risk/trade metrics aren't serialised through the worker pool.
    _metrics, best_result = _run_trial(
        best_params,
        portfolio,
        price_data,
        start,
        end,
        min_cash_pct=min_cash_pct,
        train_pct=train_pct,
        risk_config=risk_config,
        forecast_scaling=forecast_scaling,
        tax_config=tax_config,
    )

    log(
        f"  Max drawdown: {best_result.max_drawdown:.2%} | "
        f"Sharpe: {best_result.sharpe_ratio:.2f} | "
        f"Win rate: {best_result.win_rate:.2%}"
    )

    return OptimizeResult(
        best_params=best_params,
        objective=objective,
        best_objective_value=round(best_value, 4),
        best_bh_return=round(best.user_attrs["bh_return"], 4),
        best_train_return=round(best.user_attrs["train_return"], 4),
        best_test_return=round(best_result.test_return, 4),
        trials_run=len(study.trials),
        best_result=best_result,
    )


def _fold_boundaries(n_days: int, min_train_pct: float, min_test_days: int) -> list[int]:
    """Compute fold boundary indices into the sorted trading-day list.

    Returns:
        Indices ``[train_cutoff, train_cutoff + test_size, ..., n_days]``;
        fold *i* trains on days ``[0, boundaries[i])`` and tests on
        ``[boundaries[i], boundaries[i + 1])``. The last fold absorbs any
        remainder days.

    Raises:
        ValueError: If there are too few trading days for two test folds.
    """
    train_cutoff = int(n_days * min_train_pct)
    remaining = n_days - train_cutoff
    if remaining < min_test_days * 2:
        min_needed = int(min_test_days * 2 / (1 - min_train_pct))
        msg = f"Not enough data for walk-forward ({n_days} days, need ≥{min_needed})"
        raise ValueError(msg)

    n_folds = max(remaining // min_test_days, 2)
    test_size = remaining // n_folds

    boundaries = [train_cutoff + i * test_size for i in range(n_folds + 1)]
    boundaries[-1] = n_days  # last fold absorbs remainder
    return boundaries


def _aggregate_folds(
    fold_results: list[FoldResult],
    objective: Objective,
    log: Callable[[str], None],
) -> WalkForwardResult:
    """Aggregate per-fold results into a WalkForwardResult and log the summary."""
    # Per-fold test returns are already annualized, so aggregates (mean, std,
    # best/worst) compare folds of different lengths apples-to-apples.
    test_returns = [fold.test_return for fold in fold_results]
    mean_test = sum(test_returns) / len(test_returns)
    variance = sum((ret - mean_test) ** 2 for ret in test_returns) / max(len(test_returns) - 1, 1)
    std_test = variance**0.5

    # Overall OOS CAGR: compound the *raw* per-fold returns (what actually
    # happened to the equity chain) and then annualize over the full OOS span.
    # Doing this on raw values avoids the double-annualization artefact that
    # would come from chaining already-annualized fold returns.
    compounded = math.prod(1.0 + fold.test_return_raw for fold in fold_results)
    first_test_start = fold_results[0].test_start
    last_test_end = fold_results[-1].test_end
    years = (last_test_end - first_test_start).days / DAYS_PER_YEAR
    annualized = compounded ** (1.0 / years) - 1.0 if years > 0 and compounded > 0 else 0.0

    after_tax_annualized: float | None = None
    if all(fold.after_tax_test_return_raw is not None for fold in fold_results):
        after_tax_compounded = math.prod(1.0 + (fold.after_tax_test_return_raw or 0.0) for fold in fold_results)
        after_tax_annualized = (
            after_tax_compounded ** (1.0 / years) - 1.0 if years > 0 and after_tax_compounded > 0 else 0.0
        )

    winning_folds = sum(1 for ret in test_returns if ret > 0)
    best_fold = max(test_returns)
    worst_fold = min(test_returns)

    # Efficiency ratio: how much in-sample performance survives out-of-sample.
    # Both sides are annualized so the ratio is dimensionally consistent even
    # though IS windows are anchored (longer) and OOS windows are shorter.
    train_returns = [fold.train_return for fold in fold_results]
    mean_train = sum(train_returns) / len(train_returns)
    efficiency = mean_test / mean_train if mean_train != 0 else 0.0

    num_folds = len(fold_results)
    mean_dd = sum(fold.max_drawdown for fold in fold_results) / num_folds
    mean_sharpe = sum(fold.sharpe_ratio for fold in fold_results) / num_folds
    mean_sortino = sum(fold.sortino_ratio for fold in fold_results) / num_folds
    mean_wr = sum(fold.win_rate for fold in fold_results) / num_folds

    log("")
    log("Walk-forward complete")
    log(f"  Annualized OOS return (CAGR): {annualized:.2%}")
    if after_tax_annualized is not None:
        log(f"  Annualized OOS return after tax: {after_tax_annualized:.2%}")
    log(f"  Per-fold mean: {mean_test:.2%} ± {std_test:.2%}")
    log(f"  Winning folds: {winning_folds}/{num_folds} | Best: {best_fold:.2%} | Worst: {worst_fold:.2%}")
    log(f"  Efficiency ratio: {efficiency:.0%}")
    log(f"  Mean max drawdown: {mean_dd:.2%} | Mean Sharpe: {mean_sharpe:.2f}")

    return WalkForwardResult(
        folds=fold_results,
        annualized_return=round(annualized, 4),
        mean_test_return=round(mean_test, 4),
        std_test_return=round(std_test, 4),
        winning_folds=winning_folds,
        best_fold_return=round(best_fold, 4),
        worst_fold_return=round(worst_fold, 4),
        efficiency_ratio=round(efficiency, 4),
        best_params=fold_results[-1].best_params,
        total_trials=sum(fold.trials_run for fold in fold_results),
        objective=objective,
        after_tax_annualized_return=round(after_tax_annualized, 4) if after_tax_annualized is not None else None,
        mean_max_drawdown=round(mean_dd, 4),
        mean_sharpe=round(mean_sharpe, 4),
        mean_sortino=round(mean_sortino, 4),
        mean_win_rate=round(mean_wr, 4),
    )


def walk_forward_optimize(
    portfolio: PortfolioConfig,
    price_data: dict[str, pd.DataFrame],
    start: date,
    end: date,
    strategy_names: list[str] | None = None,
    n_trials: int = DEFAULT_N_TRIALS,
    min_cash_pct: float = DEFAULT_MIN_CASH_PCT,
    min_train_pct: float = WF_MIN_TRAIN_PCT,
    min_test_days: int = WF_MIN_TEST_DAYS,
    log_fn: Callable[[str], None] | None = None,
    risk_config: RiskConfig | None = None,
    forecast_scaling: ForecastScaling = "none",
    objective: Objective = DEFAULT_OBJECTIVE,
    tax_config: TaxConfig | None = None,
    seed: int = DEFAULT_SEED,
) -> WalkForwardResult:
    """Walk-forward optimisation with anchored training windows.

    Reserves *min_train_pct* of trading days as the minimum training window,
    then carves the remainder into test windows of *min_test_days* each.
    Each fold grows the training window while the test window slides forward.
    Parameters are re-optimised per fold (scored by *objective* on the
    fold's training window) so every test period is genuinely out-of-sample.

    Raises:
        ValueError: On an unknown objective, ``net`` without a tax config, or
            too few trading days for two test folds.
    """
    log = log_fn or (lambda _: None)
    _require_tax_for_net(objective, tax_config)

    names, ranges = _prepare_names_and_ranges(strategy_names, min_cash_pct, portfolio.active_ticker_count())
    trading_days = _trading_days(price_data, start, end)

    fold_boundaries = _fold_boundaries(len(trading_days), min_train_pct, min_test_days)
    n_folds = len(fold_boundaries) - 1
    test_size = fold_boundaries[1] - fold_boundaries[0]

    trials_per_fold = max(n_trials // n_folds, 10)
    max_workers = _pool_workers(trials_per_fold)

    strat_names = [name for name in names if name != ALLOCATION_KEY]
    log(f"Walk-forward optimization — {len(strat_names)} strategies, {start} to {end}")
    log(f"  objective: {objective}")
    log(f"  {n_folds} folds, ~{test_size} trading days per test window, {trials_per_fold} trials/fold")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    fold_results: list[FoldResult] = []
    prev_best_flat: dict[str, float] | None = None

    # Single pool reused across all folds — dates are passed per call.
    pool = ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_wf_init_worker,
        initargs=(portfolio, price_data, min_cash_pct, risk_config, forecast_scaling, tax_config),
    )

    try:
        for fold_idx in range(n_folds):
            # Anchored training: always starts at day 0, grows each fold.
            fold_train_start = trading_days[0]
            fold_train_end = trading_days[fold_boundaries[fold_idx] - 1]
            fold_test_start = trading_days[fold_boundaries[fold_idx]]
            fold_test_end = trading_days[fold_boundaries[fold_idx + 1] - 1]

            log(f"Fold {fold_idx + 1}/{n_folds}")
            log(f"  Train: {fold_train_start} → {fold_train_end}")
            log(f"  Test:  {fold_test_start} → {fold_test_end}")

            # --- Optimise on training window (no internal split) ---
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=seed + fold_idx),
            )

            # Warm-start: seed with previous fold's best params so Optuna
            # doesn't explore blindly when adjacent folds share most data.
            if prev_best_flat is not None:
                study.enqueue_trial(prev_best_flat)

            def submit_fold(
                params: dict[str, dict[str, float]],
                train_start: date = fold_train_start,
                train_end: date = fold_train_end,
            ) -> Future[TrialMetrics]:
                return pool.submit(_wf_trial_worker, params, train_start, train_end)

            _run_trials_batched(
                study,
                names,
                ranges,
                trials_per_fold,
                batch_size=max_workers,
                objective=objective,
                submit=submit_fold,
                log=log,
            )

            best_params: dict[str, dict[str, float]] = study.best_trial.user_attrs["params"]
            train_return_raw = study.best_trial.user_attrs["train_return"]
            objective_value = study.best_trial.value or 0.0
            prev_best_flat = dict(study.best_trial.params)

            # --- Evaluate best params on test window ---
            test_metrics, test_result = _run_trial(
                best_params,
                portfolio,
                price_data,
                fold_test_start,
                fold_test_end,
                min_cash_pct,
                enable_split=False,
                risk_config=risk_config,
                forecast_scaling=forecast_scaling,
                tax_config=tax_config,
                track_vol_contribution=False,
            )
            test_twr = test_metrics.twr

            train_days = (fold_train_end - fold_train_start).days
            test_days = (fold_test_end - fold_test_start).days
            train_ann = compute_annualized_return(train_return_raw, train_days)
            test_ann = compute_annualized_return(test_twr, test_days)
            after_tax_raw = test_result.after_tax_twr
            after_tax_ann = compute_annualized_return(after_tax_raw, test_days) if after_tax_raw is not None else None

            log(f"  Result — train: {train_ann:.2%} annualized | out-of-sample: {test_ann:.2%} annualized")

            fold_results.append(
                FoldResult(
                    fold=fold_idx + 1,
                    train_start=fold_train_start,
                    train_end=fold_train_end,
                    test_start=fold_test_start,
                    test_end=fold_test_end,
                    best_params=best_params,
                    train_return=round(train_ann, 4),
                    test_return=round(test_ann, 4),
                    train_return_raw=round(train_return_raw, 4),
                    test_return_raw=round(test_twr, 4),
                    trials_run=len(study.trials),
                    objective_value=round(objective_value, 4),
                    after_tax_test_return=round(after_tax_ann, 4) if after_tax_ann is not None else None,
                    after_tax_test_return_raw=round(after_tax_raw, 4) if after_tax_raw is not None else None,
                    max_drawdown=round(test_result.max_drawdown, 4),
                    sharpe_ratio=round(test_result.sharpe_ratio, 4),
                    sortino_ratio=round(test_result.sortino_ratio, 4),
                    win_rate=round(test_result.win_rate, 4),
                )
            )
    finally:
        pool.shutdown(wait=True)

    return _aggregate_folds(fold_results, objective, log)


def write_strategies_yaml(
    params: dict[str, dict[str, float]],
    path: str,
    min_cash_pct: float = DEFAULT_MIN_CASH_PCT,
    risk_config: RiskConfig | None = None,
    forecast_scaling: ForecastScaling = "none",
    objective: Objective | None = None,
) -> None:
    """Write optimized parameters to a strategies YAML file.

    The optimizer does not search risk knobs (policy, not tunables). When the
    user supplied a ``risk:`` block in the input, it must round-trip to the
    optimized output unchanged so the next run honors the same policy;
    otherwise the optimized YAML silently drops the user's config. The same
    holds for ``forecast_scaling``.

    Args:
        params: Per-strategy parameter dict from the optimizer (also includes
            ``ALLOCATION_KEY`` for portfolio-wide knobs).
        path: Output path.
        min_cash_pct: Preserved from the user's input config.
        risk_config: Preserved from the user's input config. When present and
            differing from defaults, emitted as a ``risk:`` block. ``None`` or
            an all-default ``RiskConfig`` is omitted.
        forecast_scaling: Preserved from the user's input config; the default
            ``"none"`` is omitted from the output.
        objective: When given, recorded as a leading comment so the file says
            what it was optimized for. A comment, never a key — the loader
            must not see it as config.
    """
    output: dict[str, object] = {}

    # Emit global allocation knobs as top-level keys
    if ALLOCATION_KEY in params:
        for key, val in params[ALLOCATION_KEY].items():
            output[key] = round(val, 4)

    # min_cash_pct is not optimized — preserve the user's configured value
    output["min_cash_pct"] = round(min_cash_pct, 4)

    # forecast_scaling is policy, not a tunable — round-trip it like the
    # risk: block, omitting the default.
    if forecast_scaling != "none":
        output["forecast_scaling"] = forecast_scaling

    risk_block = _risk_block_for_yaml(risk_config)
    if risk_block:
        output["risk"] = risk_block

    strategies = []
    for name, param_dict in params.items():
        if name == ALLOCATION_KEY:
            continue
        entry: dict[str, object] = {"name": name}
        clean_params: dict[str, object] = {}
        for key, val in param_dict.items():
            if key == "_weight":
                entry["weight"] = round(val, 4)
            elif key in INT_PARAMS:
                clean_params[key] = int(val)
            else:
                clean_params[key] = round(val, 4)
        if clean_params:
            entry["params"] = clean_params
        strategies.append(entry)

    output["strategies"] = strategies

    with open(path, "w") as handle:
        if objective is not None:
            handle.write(f"# optimized with --objective {objective}\n")
        yaml.dump(output, handle, default_flow_style=False, sort_keys=False)


def _risk_block_for_yaml(risk_config: RiskConfig | None) -> dict[str, object]:
    """Render a ``risk:`` block dict, omitting fields that match the dataclass default.

    Returns ``{}`` when ``risk_config`` is ``None`` or every field is at its
    default — there is no behavioral signal to preserve. Otherwise emits only
    the fields the user (implicitly or explicitly) set to a non-default value,
    matching the spec's "omit to disable" YAML conventions.
    """
    if risk_config is None:
        return {}
    default = RiskConfig()
    block: dict[str, object] = {}
    if risk_config.weighting != default.weighting:
        block["weighting"] = risk_config.weighting
    if risk_config.vol_lookback_days != default.vol_lookback_days:
        block["vol_lookback_days"] = int(risk_config.vol_lookback_days)
    if risk_config.vol_target is not None:
        block["vol_target"] = round(float(risk_config.vol_target), 4)
    if risk_config.drawdown_penalty is not None:
        block["drawdown_penalty"] = round(float(risk_config.drawdown_penalty), 4)
    if risk_config.drawdown_floor is not None:
        block["drawdown_floor"] = round(float(risk_config.drawdown_floor), 4)
    return block
