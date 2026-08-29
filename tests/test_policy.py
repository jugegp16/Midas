"""Tests for the deployment policy model."""

from datetime import date

import pytest

from midas.config import load_policy
from midas.models import AllocationConstraints, RiskConfig
from midas.policy import AdoptTrigger, Policy, derive_seed, input_fingerprint, parse_policy, policy_hash


def test_defaults_match_the_spec() -> None:
    policy = Policy()
    assert policy.objective == "calmar"
    assert policy.deployment == "best"
    assert (policy.budget, policy.restarts, policy.ensemble_size, policy.cadence_days) == (2000, 4, 20, 63)


def test_parse_policy_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="cadence"):
        parse_policy({"cadence": 63})


def test_parse_policy_nested_trigger() -> None:
    policy = parse_policy({"adopt_trigger": {"oos_percentile_below": 5, "drawdown_breach": False}})
    assert policy.adopt_trigger == AdoptTrigger(oos_percentile_below=5.0, drawdown_breach=False)


def test_ensemble_size_must_cover_restarts() -> None:
    with pytest.raises(ValueError, match="ensemble_size"):
        Policy(deployment="ensemble", restarts=8, ensemble_size=4)


def test_policy_hash_ignores_inert_ensemble_size_under_best() -> None:
    a = Policy(deployment="best", ensemble_size=20)
    b = Policy(deployment="best", ensemble_size=40)
    assert policy_hash(a) == policy_hash(b)
    c = Policy(deployment="ensemble", ensemble_size=20)
    d = Policy(deployment="ensemble", ensemble_size=40)
    assert policy_hash(c) != policy_hash(d)


def test_derive_seed_is_stable_and_distinct() -> None:
    s = derive_seed(42, date(2026, 8, 29), 0)
    assert s == derive_seed(42, date(2026, 8, 29), 0)  # deterministic
    assert s != derive_seed(42, date(2026, 8, 29), 1)  # restart-distinct
    assert s != derive_seed(42, date(2026, 8, 28), 0)  # date-distinct
    assert s != derive_seed(7, date(2026, 8, 29), 0)  # base-distinct
    assert 0 <= s < 2**32


def test_input_fingerprint_moves_with_every_input() -> None:
    base = dict(
        policy=Policy(),
        tickers=["AAA", "BBB"],
        constraints=AllocationConstraints(),
        exit_params={"StopLoss": {"loss_threshold": 0.1}},
        risk_config=RiskConfig(),
        min_cash_pct=0.05,
        forecast_scaling="quantile",
        tax_config=None,
    )
    ref = input_fingerprint(**base)
    assert input_fingerprint(**{**base, "tickers": ["BBB", "AAA"]}) == ref  # order-insensitive
    assert input_fingerprint(**{**base, "tickers": ["AAA"]}) != ref
    assert input_fingerprint(**{**base, "min_cash_pct": 0.06}) != ref
    assert input_fingerprint(**{**base, "exit_params": {}}) != ref


def test_load_policy_roundtrip(tmp_path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text("strategies:\n  - name: MeanReversion\npolicy:\n  objective: gross\n  budget: 100\n  restarts: 2\n")
    policy = load_policy(path)
    assert policy is not None and policy.objective == "gross" and policy.budget == 100
    path.write_text("strategies:\n  - name: MeanReversion\n")
    assert load_policy(path) is None
