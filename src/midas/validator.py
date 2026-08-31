"""Policy validation: execute the deployment policy exactly as live will.

Cadence-driven folds, one fit per fold via ``fit_as_of``, degradation-gated
adoption evaluated over full daily paths, and a portfolio that carries
across fold boundaries — emitting the baselines artifact the monitor and
sweep consume.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import date

import pandas as pd

from midas.allocator import Allocator
from midas.backtest import BacktestEngine
from midas.baselines import Baselines, FoldRecord
from midas.fitter import FitResult, fit_as_of
from midas.metrics import compute_annualized_return
from midas.models import (
    DEFAULT_MIN_CASH_PCT,
    AllocationConstraints,
    ForecastScaling,
    PortfolioConfig,
    RiskConfig,
    TaxConfig,
)
from midas.monitor import offset_fold_rank
from midas.optimizer import _instantiate_strategies
from midas.order_sizer import OrderSizer
from midas.policy import AdoptTrigger, Policy, input_fingerprint, policy_hash

# Arming thresholds: an arm stays silent until enough validation history
# exists to give it meaning — mirroring what live could have known at the
# same point in time.
# Three priors, not one: a single fold's worst drawdown is a trivially low
# bar, and Experiment 1 measured the resulting adoptions at ~50% of folds —
# past the spec's ~1-in-5 false-alarm kill criterion.
DRAWDOWN_ARM_MIN_FOLDS = 3
PERCENTILE_ARM_MIN_FOLDS = 5


def trigger_fired(
    prior_folds: Sequence[FoldRecord],
    current_daily_returns: Sequence[float],
    trigger: AdoptTrigger,
) -> bool:
    """Whether the degradation trigger fired at any point along *current* path.

    Latched semantics: the answer is the max over the path's day offsets,
    because live evaluates nightly and a mid-window breach counts even if
    the window later recovers.

    Arms:
        drawdown (>= 1 prior fold): the path's drawdown from its window
            peak exceeds the worst prior-fold drawdown.
        percentile (>= 5 prior folds): the cumulative return at some day
            offset ranks below the configured percentile of prior folds'
            cumulative returns at the same offset — the identical midrank
            the monitor prints, so the trigger fires on the number the
            operator reads, never a different one.
    """
    if trigger.disabled or not current_daily_returns:
        return False

    if trigger.drawdown_breach and len(prior_folds) >= DRAWDOWN_ARM_MIN_FOLDS:
        worst_prior = max(fold.drawdown for fold in prior_folds)
        if _max_drawdown_along(current_daily_returns) > worst_prior:
            return True

    if trigger.oos_percentile_below > 0 and len(prior_folds) >= PERCENTILE_ARM_MIN_FOLDS:
        prior_cumulative = [_cumulative(fold.daily_returns) for fold in prior_folds]
        current_cumulative = _cumulative(current_daily_returns)
        for offset, value in enumerate(current_cumulative):
            long_enough = sum(1 for path in prior_cumulative if len(path) > offset)
            if long_enough < PERCENTILE_ARM_MIN_FOLDS:
                continue
            rank = offset_fold_rank(prior_cumulative, offset, value)
            if rank is not None and rank < trigger.oos_percentile_below:
                return True

    return False


def _cumulative(daily_returns: Sequence[float]) -> list[float]:
    values: list[float] = []
    value = 1.0
    for ret in daily_returns:
        value *= 1.0 + ret
        values.append(value - 1.0)
    return values


def _max_drawdown_along(daily_returns: Sequence[float]) -> float:
    value = 1.0
    peak = 1.0
    worst = 0.0
    for ret in daily_returns:
        value *= 1.0 + ret
        peak = max(peak, value)
        worst = max(worst, (peak - value) / peak)
    return worst


def validate_policy(
    portfolio: PortfolioConfig,
    price_data: dict[str, pd.DataFrame],
    start: date,
    end: date,
    policy: Policy,
    *,
    constraints: AllocationConstraints,
    exit_params: dict[str, dict[str, float | int | str]],
    risk_config: RiskConfig | None = None,
    min_cash_pct: float = DEFAULT_MIN_CASH_PCT,
    forecast_scaling: ForecastScaling = "none",
    tax_config: TaxConfig | None = None,
    strategy_names: list[str] | None = None,
    min_train_pct: float = 0.60,
    log_fn: Callable[[str], None] | None = None,
) -> Baselines:
    """Execute *policy* over ``[start, end]`` exactly as live would run it.

    Folds are ``policy.cadence_days`` long (the last absorbs the remainder)
    after an initial training reserve. Each fold fits via ``fit_as_of`` with
    the deployed members as incumbent, adopts the challenger only when the
    degradation trigger fired along the previous fold's path (fold 0 always
    adopts — there is nothing to keep), and evaluates on a portfolio carried
    across fold boundaries, so parameter switches pay their real turnover
    and tax cost and no fold ramps in from cash.

    Raises:
        ValueError: When the range leaves no room for a single test fold.
    """
    log = log_fn or (lambda _msg: None)
    all_days: set[date] = set()
    for frame in price_data.values():
        all_days.update(d for d in frame.index if start <= d <= end)
    days = sorted(all_days)
    initial_train = int(len(days) * min_train_pct)
    if initial_train >= len(days) - 1:
        msg = f"no room for a test fold: {len(days)} trading days at min_train_pct={min_train_pct}"
        raise ValueError(msg)
    boundaries = list(range(initial_train, len(days), policy.cadence_days))
    if len(days) - boundaries[-1] < policy.cadence_days and len(boundaries) > 1:
        boundaries.pop()  # last fold absorbs the remainder

    numeric_exits = {name: {k: float(v) for k, v in block.items()} for name, block in exit_params.items()}
    _, exit_rules = _instantiate_strategies(numeric_exits)

    deployed: FitResult | None = None
    carried = None
    completed: list[FoldRecord] = []
    for fold_number, boundary in enumerate(boundaries, start=1):
        test_start = days[boundary]
        end_boundary = boundary + policy.cadence_days
        test_end = days[min(end_boundary, len(days)) - 1] if fold_number < len(boundaries) else days[-1]
        log(f"Fold {fold_number}/{len(boundaries)}: test {test_start} → {test_end}")

        if deployed is not None and policy.adopt_trigger.disabled:
            # Frozen policy: a challenger can never be adopted, so the fit
            # would be discarded unread — skip the search entirely.
            challenger = deployed
        else:
            challenger = fit_as_of(
                portfolio,
                price_data,
                test_start,
                policy,
                constraints=constraints,
                exit_params=exit_params,
                risk_config=risk_config,
                min_cash_pct=min_cash_pct,
                forecast_scaling=forecast_scaling,
                tax_config=tax_config,
                incumbent=deployed,
                strategy_names=strategy_names,
                log_fn=log_fn,
            )
        adopted = deployed is None or trigger_fired(completed[:-1], completed[-1].daily_returns, policy.adopt_trigger)
        if adopted:
            deployed = challenger
        assert deployed is not None  # fold 0 always adopts

        members = [_instantiate_strategies(member)[0] for member in deployed.members]
        allocator = Allocator(
            [], constraints, portfolio.active_ticker_count(), risk_config=risk_config, members=members
        )
        engine = BacktestEngine(
            allocator=allocator,
            order_sizer=OrderSizer(),
            exit_rules=exit_rules,
            constraints=constraints,
            enable_split=False,
            tax_config=tax_config,
            track_vol_contribution=False,
        )
        result = engine.run(portfolio, price_data, test_start, test_end, carried=carried)
        carried = result.end_state

        raw = result.twr if result.twr is not None else 0.0
        window_days = (test_end - test_start).days
        completed.append(
            FoldRecord(
                fold=fold_number,
                test_start=test_start,
                test_end=test_end,
                daily_returns=list(result.daily_returns),
                return_raw=round(raw, 6),
                return_annualized=round(compute_annualized_return(raw, window_days), 6),
                drawdown=round(_max_drawdown_along(result.daily_returns), 6),
                after_tax_return_raw=(round(result.after_tax_twr, 6) if result.after_tax_twr is not None else None),
                adopted=adopted,
            )
        )

    compounded = math.prod(1.0 + fold.return_raw for fold in completed)
    oos_days = (completed[-1].test_end - completed[0].test_start).days
    years = oos_days / 365.25
    aggregate = compounded ** (1.0 / years) - 1.0 if years > 0 and compounded > 0 else 0.0

    return Baselines(
        folds=completed,
        aggregate_oos_cagr=round(aggregate, 6),
        policy_hash=policy_hash(policy),
        input_hash=input_fingerprint(
            policy=policy,
            tickers=[h.ticker for h in portfolio.holdings],
            constraints=constraints,
            exit_params=exit_params,
            risk_config=risk_config,
            min_cash_pct=min_cash_pct,
            forecast_scaling=forecast_scaling,
            tax_config=tax_config,
            cash_infusion=portfolio.cash_infusion,
            trading_restrictions=portfolio.trading_restrictions,
            strategy_names=strategy_names or [],
        ),
        validation_start=days[0],
        validation_end=days[-1],
        cadence_days=policy.cadence_days,
    )
