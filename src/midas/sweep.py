"""Policy sweeps: compare variants with statistics that respect the noise.

The measured lesson behind this module: sampler seed alone moved OOS CAGR
by 3-8x more than the objective choice did, so any comparison that lacks
an error bar is a comparison of dice rolls. The sweep replicates across
seeds, bootstraps the OOS path, deflates the Sharpe for search intensity,
and refuses to print an unqualified verdict from a single portfolio.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from statistics import NormalDist

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
