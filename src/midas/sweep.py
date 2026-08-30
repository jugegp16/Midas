"""Policy sweeps: compare variants with statistics that respect the noise.

The measured lesson behind this module: sampler seed alone moved OOS CAGR
by 3-8x more than the objective choice did, so any comparison that lacks
an error bar is a comparison of dice rolls. The sweep replicates across
seeds, bootstraps the OOS path, deflates the Sharpe for search intensity,
and refuses to print an unqualified verdict from a single portfolio.
"""

from __future__ import annotations

import dataclasses
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from statistics import NormalDist

import pandas as pd

from midas.baselines import Baselines
from midas.metrics import compute_sharpe
from midas.models import (
    DEFAULT_MIN_CASH_PCT,
    AllocationConstraints,
    ForecastScaling,
    PortfolioConfig,
    RiskConfig,
    TaxConfig,
)
from midas.policy import AdoptTrigger, Policy
from midas.validator import validate_policy

TRADING_DAYS_PER_YEAR = 252
EULER_MASCHERONI = 0.5772156649015329


def stationary_bootstrap(
    daily_returns: Sequence[float],
    *,
    n_resamples: int = 1000,
    mean_block: int = 21,
    seed: int = 0,
) -> list[float]:
    """Politis-Romano stationary bootstrap of an OOS daily-return path.

    Resamples wrap-around blocks with geometric lengths (mean
    ``mean_block`` bars) and returns one annualized return per resample —
    the uncertainty of the aggregate OOS CAGR under serial dependence,
    which an i.i.d. bootstrap would understate.
    """
    n = len(daily_returns)
    if n < 2:
        return []
    rng = random.Random(seed)
    restart_p = 1.0 / mean_block
    out: list[float] = []
    for _ in range(n_resamples):
        idx = rng.randrange(n)
        growth = 1.0
        for _step in range(n):
            growth *= 1.0 + daily_returns[idx]
            idx = rng.randrange(n) if rng.random() < restart_p else (idx + 1) % n
        out.append(growth ** (TRADING_DAYS_PER_YEAR / n) - 1.0)
    return out


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_trials: int,
    sharpe_variance: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Bailey / Lopez de Prado DSR: P[the observed Sharpe beats noise's max].

    ``n_trials`` is how many candidates were effectively tried before the
    reported one was selected; ``sharpe_variance`` is the variance of
    Sharpe estimates across those candidates. The callers here proxy both
    (trial counts from the policy, variance from per-fold OOS Sharpe) and
    say so in their output — the proxy is stated, never hidden.
    """
    if n_trials < 1 or n_obs < 2:
        return 0.5
    normal = NormalDist()
    spread = math.sqrt(max(sharpe_variance, 1e-12))
    expected_max = spread * (
        (1.0 - EULER_MASCHERONI) * normal.inv_cdf(1.0 - 1.0 / max(n_trials, 2))
        + EULER_MASCHERONI * normal.inv_cdf(1.0 - 1.0 / (max(n_trials, 2) * math.e))
    )
    denom = 1.0 - skew * observed_sharpe + (kurtosis - 1.0) / 4.0 * observed_sharpe**2
    if denom <= 0:
        return 0.0
    z = (observed_sharpe - expected_max) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return min(max(normal.cdf(z), 0.0), 1.0)


@dataclass(frozen=True)
class SweepCell:
    """One validation run: a variant on a portfolio under one seed base."""

    variant: str
    portfolio: str
    seed_base: int
    aggregate_oos_cagr: float
    oos_sharpe: float
    daily_returns: list[float]
    n_folds: int


@dataclass(frozen=True)
class SweepReport:
    """All cells of a sweep plus what was withheld from it."""

    cells: list[SweepCell]
    holdout_trimmed_to: date | None = None

    def summary_lines(self) -> list[str]:
        """Per-(variant, portfolio) means with noise floors and a verdict."""
        lines: list[str] = []
        portfolios = sorted({cell.portfolio for cell in self.cells})
        variants = sorted({cell.variant for cell in self.cells})
        single_portfolio = len(portfolios) == 1
        widest_floor = 0.0
        means: dict[tuple[str, str], float] = {}
        for portfolio in portfolios:
            for variant in variants:
                cells = [c for c in self.cells if c.portfolio == portfolio and c.variant == variant]
                if not cells:
                    continue
                cagrs = [c.aggregate_oos_cagr for c in cells]
                mean = sum(cagrs) / len(cagrs)
                spread = max(cagrs) - min(cagrs) if len(cagrs) > 1 else 0.0
                widest_floor = max(widest_floor, spread)
                means[(portfolio, variant)] = mean
                # Each seed's path is resampled on its own and the resampled
                # CAGRs pooled — concatenating first would narrow the CI by
                # sqrt(seeds) and average distinct basins into one blur.
                boot: list[float] = []
                for cell_idx, cell in enumerate(cells):
                    boot.extend(stationary_bootstrap(cell.daily_returns, n_resamples=500, seed=cell_idx))
                ci = ""
                if boot:
                    boot.sort()
                    lo = boot[int(0.025 * len(boot))]
                    hi = boot[int(0.975 * len(boot)) - 1]
                    ci = f"  bootstrap95% [{lo:+.2%}, {hi:+.2%}]"
                path = [r for c in cells for r in c.daily_returns]
                # The DSR z-statistic works in per-period units: n_obs counts
                # daily observations, so the annualized Sharpe and its
                # cross-seed variance both convert to daily scale.
                sharpe_var = _variance([c.oos_sharpe for c in cells]) / TRADING_DAYS_PER_YEAR
                n_trials = max(1, sum(c.n_folds for c in cells))
                skew, kurtosis = _sample_moments(path)
                dsr = deflated_sharpe_ratio(
                    sum(c.oos_sharpe for c in cells) / len(cells) / math.sqrt(TRADING_DAYS_PER_YEAR),
                    n_trials=n_trials,
                    sharpe_variance=max(sharpe_var, 0.01 / TRADING_DAYS_PER_YEAR),
                    n_obs=max(len(path), 2),
                    skew=skew,
                    kurtosis=kurtosis,
                )
                lines.append(
                    f"{portfolio}/{variant}: mean OOS CAGR {mean:+.2%} across {len(cells)} seed(s), "
                    f"spread {spread:.2%}{ci}  DSR~{dsr:.2f} (proxy: trials from folds, "
                    f"SR var from seeds)"
                )
        for portfolio in portfolios:
            variant_means = [means[(portfolio, v)] for v in variants if (portfolio, v) in means]
            if len(variant_means) > 1:
                diff = max(variant_means) - min(variant_means)
                qualifier = " (on these windows)" if single_portfolio else ""
                if diff < widest_floor:
                    lines.append(
                        f"{portfolio}: variant spread {diff:.2%} vs seed noise {widest_floor:.2%} — "
                        f"not resolvable at this budget{qualifier}"
                    )
                else:
                    lines.append(
                        f"{portfolio}: variant spread {diff:.2%} exceeds seed noise {widest_floor:.2%}{qualifier}"
                    )
        return lines


def make_variants(policy: Policy, vary: str, values: Sequence[str]) -> list[tuple[str, Policy]]:
    """Build named policy variants along one field.

    Raises:
        ValueError: When *vary* is not a policy field (the message lists
            the valid ones) or a value fails the field's parsing.
    """
    field_names = {f.name for f in dataclasses.fields(Policy)}
    if vary not in field_names:
        msg = f"cannot vary {vary!r}; policy fields: {sorted(field_names)} (e.g. objective)"
        raise ValueError(msg)
    if not values:
        return [("base", policy)]
    int_fields = {"budget", "restarts", "base_seed", "ensemble_size", "cadence_days"}
    variants: list[tuple[str, Policy]] = []
    for value in values:
        parsed: object
        if vary == "adopt_trigger":
            # The axis for the hold-vs-refit experiment: "none" freezes the
            # policy (fit once, hold), "default" keeps the standard trigger.
            if value == "none":
                parsed = AdoptTrigger.never()
            elif value == "default":
                parsed = AdoptTrigger()
            else:
                msg = f"adopt_trigger variant must be 'none' or 'default', got {value!r}"
                raise ValueError(msg)
        elif vary in int_fields:
            parsed = int(value)
        else:
            parsed = value
        # dataclasses.replace re-runs Policy.__post_init__, so invalid values
        # fail loudly there; the dict indirection is untypeable for mypy.
        replaced = dataclasses.replace(policy, **{vary: parsed})  # type: ignore[arg-type]
        variants.append((str(value), replaced))
    return variants


def run_sweep(
    portfolios: Mapping[str, PortfolioConfig],
    price_data_by_portfolio: Mapping[str, dict[str, pd.DataFrame]],
    start: date,
    end: date,
    policy: Policy,
    vary: str,
    values: Sequence[str],
    seeds: int,
    *,
    constraints: AllocationConstraints,
    exit_params: dict[str, dict[str, float | int | str]],
    risk_config: RiskConfig | None = None,
    min_cash_pct: float = DEFAULT_MIN_CASH_PCT,
    forecast_scaling: ForecastScaling = "none",
    tax_config: TaxConfig | None = None,
    strategy_names: list[str] | None = None,
    log_fn: Callable[[str], None] | None = None,
    validate: Callable[..., Baselines] = validate_policy,
    holdout_trimmed_to: date | None = None,
) -> SweepReport:
    """Fan the validator over variants x portfolios x seed replications.

    Replication *i* runs the policy with ``base_seed + i``; restart seeds
    then hash from that base with each fold's as_of, so replications never
    collide with restarts.
    """
    log = log_fn or (lambda _msg: None)
    cells: list[SweepCell] = []
    for variant_name, variant in make_variants(policy, vary, values):
        for portfolio_name, portfolio in portfolios.items():
            for i in range(seeds):
                seeded = dataclasses.replace(variant, base_seed=variant.base_seed + i)
                log(f"sweep: {variant_name} on {portfolio_name}, seed base {seeded.base_seed}")
                baselines = validate(
                    portfolio,
                    price_data_by_portfolio[portfolio_name],
                    start,
                    end,
                    seeded,
                    constraints=constraints,
                    exit_params=exit_params,
                    risk_config=risk_config,
                    min_cash_pct=min_cash_pct,
                    forecast_scaling=forecast_scaling,
                    tax_config=tax_config,
                    strategy_names=strategy_names,
                )
                path = [r for fold in baselines.folds for r in fold.daily_returns]
                cells.append(
                    SweepCell(
                        variant=variant_name,
                        portfolio=portfolio_name,
                        seed_base=seeded.base_seed,
                        aggregate_oos_cagr=baselines.aggregate_oos_cagr,
                        oos_sharpe=compute_sharpe(path),
                        daily_returns=path,
                        n_folds=len(baselines.folds),
                    )
                )
    return SweepReport(cells=cells, holdout_trimmed_to=holdout_trimmed_to)


def _sample_moments(values: Sequence[float]) -> tuple[float, float]:
    """Sample skewness and kurtosis (non-excess) of a return path.

    Returns:
        ``(0.0, 3.0)`` — the normal case — when the path is too short or
        degenerate to estimate them.
    """
    n = len(values)
    if n < 4:
        return 0.0, 3.0
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    if m2 <= 0:
        return 0.0, 3.0
    m3 = sum((v - mean) ** 3 for v in values) / n
    m4 = sum((v - mean) ** 4 for v in values) / n
    return m3 / m2**1.5, m4 / m2**2


def _variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / (len(values) - 1)
