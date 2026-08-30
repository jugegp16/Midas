"""Tests for fit_as_of."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
from conftest import make_price_series

from midas.fitter import FitResult, fit_as_of
from midas.models import AllocationConstraints, Holding, PortfolioConfig
from midas.policy import Policy


def _data() -> tuple[PortfolioConfig, dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(5)
    prices = make_price_series(date(2023, 1, 2), 240, 100.0, list(rng.normal(0.0005, 0.012, 240)), name="TEST")
    portfolio = PortfolioConfig(holdings=[Holding(ticker="TEST", shares=10, cost_basis=100.0)], available_cash=2000.0)
    return portfolio, {"TEST": prices}


SMALL = Policy(objective="gross", budget=6, restarts=2, base_seed=42, deployment="best")


def test_fit_is_deterministic() -> None:
    portfolio, price_data = _data()
    kwargs = dict(constraints=AllocationConstraints(), exit_params={}, strategy_names=["MeanReversion"])
    a = fit_as_of(portfolio, price_data, date(2023, 10, 2), SMALL, **kwargs)
    b = fit_as_of(portfolio, price_data, date(2023, 10, 2), SMALL, **kwargs)
    assert a.members == b.members
    assert a.restart_bests == b.restart_bests
    assert a.policy_hash == b.policy_hash and a.input_hash == b.input_hash


def test_as_of_is_exclusive(monkeypatch) -> None:
    """A fit at bar t must not see bar t: every search window ends the day
    before as_of. (Two different as_of dates legitimately differ even on
    identical data, because seeds derive from as_of — so exclusivity is
    pinned at the seam, not by comparing runs.)"""
    import midas.fitter as fitter_module

    captured: list[date] = []
    real_optimize = fitter_module.optimize

    def spy(*args, **kwargs):
        captured.append(kwargs["end"])
        return real_optimize(*args, **kwargs)

    monkeypatch.setattr(fitter_module, "optimize", spy)
    portfolio, price_data = _data()
    bars = sorted(price_data["TEST"].index)
    as_of = bars[100]
    fit_as_of(
        portfolio,
        price_data,
        as_of,
        SMALL,
        constraints=AllocationConstraints(),
        exit_params={},
        strategy_names=["MeanReversion"],
    )
    assert captured and all(end == as_of - timedelta(days=1) for end in captured)


def test_best_mode_selects_the_best_restart() -> None:
    portfolio, price_data = _data()
    result = fit_as_of(
        portfolio,
        price_data,
        date(2023, 10, 2),
        SMALL,
        constraints=AllocationConstraints(),
        exit_params={},
        strategy_names=["MeanReversion"],
    )
    assert isinstance(result, FitResult)
    assert len(result.members) == 1
    assert len(result.restart_bests) == 2
    assert result.member_scores == [max(result.restart_bests)]


def test_incumbent_members_are_enqueued_into_restart_zero_only(monkeypatch) -> None:
    """Observed via the seam: capture optimize() calls and inspect enqueue."""
    import midas.fitter as fitter_module

    calls = []
    real_optimize = fitter_module.optimize

    def spy(*args, **kwargs):
        calls.append(kwargs.get("enqueue"))
        return real_optimize(*args, **kwargs)

    monkeypatch.setattr(fitter_module, "optimize", spy)
    portfolio, price_data = _data()
    kwargs = dict(constraints=AllocationConstraints(), exit_params={}, strategy_names=["MeanReversion"])
    first = fit_as_of(portfolio, price_data, date(2023, 8, 1), SMALL, **kwargs)
    calls.clear()
    fit_as_of(portfolio, price_data, date(2023, 10, 2), SMALL, incumbent=first, **kwargs)
    assert calls[0] is not None and len(calls[0]) == 1  # restart 0: the incumbent member
    assert all(c is None for c in calls[1:])  # other restarts stay independent


def test_ensemble_mode_returns_stratified_members() -> None:
    portfolio, price_data = _data()
    policy = Policy(objective="gross", budget=8, restarts=2, ensemble_size=4, deployment="ensemble")
    result = fit_as_of(
        portfolio,
        price_data,
        date(2023, 10, 2),
        policy,
        constraints=AllocationConstraints(),
        exit_params={},
        strategy_names=["MeanReversion"],
    )
    assert 1 <= len(result.members) <= 4
    assert len(result.member_scores) == len(result.members)
    assert len({str(m) for m in result.members}) == len(result.members)  # distinct
    assert result.member_scores == sorted(result.member_scores, reverse=True)


def test_ensemble_dedupe_collapses_identical_behavior(monkeypatch) -> None:
    """Force every candidate to one behavioral key: at most one member may
    survive per restart stratum."""
    import midas.fitter as fitter_module

    monkeypatch.setattr(fitter_module, "behavioral_key", lambda series: ("same",))
    portfolio, price_data = _data()
    policy = Policy(objective="gross", budget=8, restarts=2, ensemble_size=4, deployment="ensemble")
    result = fit_as_of(
        portfolio,
        price_data,
        date(2023, 10, 2),
        policy,
        constraints=AllocationConstraints(),
        exit_params={},
        strategy_names=["MeanReversion"],
    )
    assert len(result.members) == 1  # one global behavior -> one member total


def test_ensemble_mode_is_deterministic() -> None:
    portfolio, price_data = _data()
    policy = Policy(objective="gross", budget=8, restarts=2, ensemble_size=4, deployment="ensemble")
    kwargs = dict(constraints=AllocationConstraints(), exit_params={}, strategy_names=["MeanReversion"])
    a = fit_as_of(portfolio, price_data, date(2023, 10, 2), policy, **kwargs)
    b = fit_as_of(portfolio, price_data, date(2023, 10, 2), policy, **kwargs)
    assert a.members == b.members and a.member_scores == b.member_scores


def test_ensemble_underfill_is_reported() -> None:
    """Fewer survivors than ensemble_size must be said out loud, not implied.

    The operator reads "20 members" in the policy; if dedupe leaves 3, the
    deployed artifact is a much less diversified object than configured.
    """
    import pytest

    portfolio, price_data = _data()
    policy = Policy(objective="sharpe", budget=2, restarts=2, deployment="ensemble", ensemble_size=6)
    logs: list[str] = []
    result = fit_as_of(
        portfolio,
        price_data,
        date(2023, 10, 2),
        policy,
        constraints=AllocationConstraints(),
        exit_params={},
        strategy_names=["MeanReversion"],
        log_fn=logs.append,
    )
    if len(result.members) < policy.ensemble_size:
        assert any("ensemble under-filled" in line for line in logs), logs
    else:
        pytest.skip("dedupe kept ensemble full; under-fill not reachable with this fixture")
