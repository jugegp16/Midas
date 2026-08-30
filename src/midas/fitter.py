"""fit_as_of: the one code path that produces deployable members.

The walk-forward validator calls this per fold and live's scheduled
re-fit calls it going forward — same budget, same restarts, same
selection, same seeds. Coherence between validation and deployment lives
or dies here.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from midas.allocator import Allocator
from midas.backtest import BacktestEngine
from midas.ensemble import behavioral_key
from midas.models import (
    DEFAULT_MIN_CASH_PCT,
    AllocationConstraints,
    ForecastScaling,
    PortfolioConfig,
    RiskConfig,
    TaxConfig,
)
from midas.optimizer import OptimizeResult, _instantiate_strategies, flatten_params, optimize
from midas.order_sizer import OrderSizer
from midas.policy import Policy, derive_seed, input_fingerprint, policy_hash


@dataclass(frozen=True)
class FitResult:
    """Deployable members plus the provenance needed to trust them later."""

    members: list[dict[str, dict[str, float]]]
    member_scores: list[float]
    restart_bests: list[float]
    as_of: date
    policy_hash: str
    input_hash: str


def fit_as_of(
    portfolio: PortfolioConfig,
    price_data: dict[str, pd.DataFrame],
    as_of: date,
    policy: Policy,
    *,
    constraints: AllocationConstraints,
    exit_params: dict[str, dict[str, float | int | str]],
    risk_config: RiskConfig | None = None,
    min_cash_pct: float = DEFAULT_MIN_CASH_PCT,
    forecast_scaling: ForecastScaling = "none",
    tax_config: TaxConfig | None = None,
    incumbent: FitResult | None = None,
    strategy_names: list[str] | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> FitResult:
    """Fit deployable members on data strictly before *as_of*.

    Runs ``policy.restarts`` independent seeded searches; the incumbent's
    members are enqueued into restart 0 only, so the remaining restarts
    stay independent explorations of the landscape. Selection follows
    ``policy.deployment``: the single best trial across restarts, or a
    stratified, behaviorally deduplicated top-K/R ensemble.
    """
    start = min(min(df.index) for df in price_data.values())
    end = as_of - timedelta(days=1)
    enqueue = [flatten_params(member) for member in incumbent.members] if incumbent is not None else None
    record_top = policy.ensemble_size if policy.deployment == "ensemble" else 0

    results = []
    for restart in range(policy.restarts):
        results.append(
            optimize(
                portfolio=portfolio,
                price_data=price_data,
                start=start,
                end=end,
                strategy_names=strategy_names,
                n_trials=policy.budget,
                min_cash_pct=min_cash_pct,
                train_pct=1.0,
                log_fn=log_fn,
                risk_config=risk_config,
                forecast_scaling=forecast_scaling,
                objective=policy.objective,
                tax_config=tax_config,
                seed=derive_seed(policy.base_seed, as_of, restart),
                exit_params=exit_params,
                constraints=constraints,
                enqueue=enqueue if restart == 0 else None,
                record_top=record_top,
            )
        )

    restart_bests = [r.best_objective_value for r in results]
    if policy.deployment == "best":
        best = max(results, key=lambda r: r.best_objective_value)
        members = [best.best_params]
        member_scores = [best.best_objective_value]
    else:
        members, member_scores = _select_ensemble(
            results,
            policy,
            portfolio,
            price_data,
            start,
            end,
            constraints=constraints,
            exit_params=exit_params,
            risk_config=risk_config,
            min_cash_pct=min_cash_pct,
            forecast_scaling=forecast_scaling,
            tax_config=tax_config,
        )
        if len(members) < policy.ensemble_size and log_fn is not None:
            log_fn(
                f"ensemble under-filled: {len(members)} distinct member(s) survived dedupe "
                f"(policy asks for {policy.ensemble_size}) — the deployment is less diversified than configured"
            )

    return FitResult(
        members=members,
        member_scores=member_scores,
        restart_bests=restart_bests,
        as_of=as_of,
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
        ),
    )


def _select_ensemble(
    results: list[OptimizeResult],
    policy: Policy,
    portfolio: PortfolioConfig,
    price_data: dict[str, pd.DataFrame],
    start: date,
    end: date,
    *,
    constraints: AllocationConstraints,
    exit_params: dict[str, dict[str, float | int | str]],
    risk_config: RiskConfig | None,
    min_cash_pct: float,
    forecast_scaling: ForecastScaling,
    tax_config: TaxConfig | None,
) -> tuple[list[dict[str, dict[str, float]]], list[float]]:
    """Stratified top-K/R across restarts, deduplicated by behavior.

    Each restart contributes at most ceil(K/R) members, walked best-first;
    a candidate whose training-window target series matches one already
    selected (anywhere) is skipped — a pooled top-K from TPE trials would
    be near-duplicates from the winning basin.
    """
    per_stratum = math.ceil(policy.ensemble_size / policy.restarts)
    seen: set[tuple[tuple[tuple[str, float], ...], ...]] = set()
    selected: list[tuple[float, dict[str, dict[str, float]]]] = []
    for result in results:
        taken = 0
        for score, params in result.top_trials:
            if taken >= per_stratum or len(selected) >= policy.ensemble_size:
                break
            key = behavioral_key(
                _target_series(
                    params,
                    portfolio,
                    price_data,
                    start,
                    end,
                    constraints=constraints,
                    exit_params=exit_params,
                    risk_config=risk_config,
                    min_cash_pct=min_cash_pct,
                    forecast_scaling=forecast_scaling,
                    tax_config=tax_config,
                )
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append((score, params))
            taken += 1
    selected.sort(key=lambda item: -item[0])
    return [params for _score, params in selected], [score for score, _params in selected]


def _target_series(
    params: dict[str, dict[str, float]],
    portfolio: PortfolioConfig,
    price_data: dict[str, pd.DataFrame],
    start: date,
    end: date,
    *,
    constraints: AllocationConstraints,
    exit_params: dict[str, dict[str, float | int | str]],
    risk_config: RiskConfig | None,
    min_cash_pct: float,
    forecast_scaling: ForecastScaling,
    tax_config: TaxConfig | None,
) -> list[dict[str, float]]:
    """One candidate member's per-bar targets over the training window."""
    entries, _ = _instantiate_strategies(params)
    numeric_exits = {name: {k: float(v) for k, v in block.items()} for name, block in exit_params.items()}
    _, exits = _instantiate_strategies(numeric_exits)
    allocator = Allocator(entries, constraints, portfolio.active_ticker_count(), risk_config=risk_config)
    engine = BacktestEngine(
        allocator=allocator,
        order_sizer=OrderSizer(),
        exit_rules=exits,
        constraints=constraints,
        enable_split=False,
        tax_config=tax_config,
        track_vol_contribution=False,
        record_targets=True,
    )
    result = engine.run(portfolio, price_data, start, end)
    return result.target_series
