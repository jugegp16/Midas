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
        cash_infusion=None,
        trading_restrictions=None,
        strategy_names=["MeanReversion"],
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


class TestFrozenTrigger:
    """adopt_trigger: none expresses the hold-forever (frozen) policy."""

    def test_parse_none_disables_both_arms(self) -> None:
        policy = parse_policy({"objective": "sharpe", "adopt_trigger": "none"})
        assert policy.adopt_trigger.disabled
        assert not policy.adopt_trigger.drawdown_breach
        assert policy.adopt_trigger.oos_percentile_below == 0.0

    def test_parse_yaml_null_also_disables(self) -> None:
        policy = parse_policy({"objective": "sharpe", "adopt_trigger": None})
        assert policy.adopt_trigger.disabled

    def test_default_trigger_is_not_disabled(self) -> None:
        assert not parse_policy({"objective": "sharpe"}).adopt_trigger.disabled


class TestFingerprintCoversPortfolioBehavior:
    """Anything that changes a fit's outcome must change the fingerprint."""

    def _base_kwargs(self) -> dict:
        from midas.models import AllocationConstraints

        return dict(
            policy=Policy(objective="sharpe"),
            tickers=["AAPL"],
            constraints=AllocationConstraints(),
            exit_params=None,
            risk_config=None,
            min_cash_pct=0.05,
            forecast_scaling="linear",
            tax_config=None,
            cash_infusion=None,
            trading_restrictions=None,
            strategy_names=["MeanReversion"],
        )

    def test_cash_infusion_changes_hash(self) -> None:
        from datetime import date

        from midas.models import CashInfusion

        base = input_fingerprint(**self._base_kwargs())
        kwargs = self._base_kwargs()
        kwargs["cash_infusion"] = CashInfusion(amount=500.0, frequency="monthly", next_date=date(2026, 1, 2))
        assert input_fingerprint(**kwargs) != base

    def test_trading_restrictions_change_hash(self) -> None:
        from midas.models import TradingRestrictions

        base = input_fingerprint(**self._base_kwargs())
        kwargs = self._base_kwargs()
        kwargs["trading_restrictions"] = TradingRestrictions(round_trip_days=30)
        assert input_fingerprint(**kwargs) != base


def test_parse_policy_rejects_non_integral_counts() -> None:
    """budget: 2.5 must fail loudly, never silently truncate to 2."""
    with pytest.raises(ValueError, match="whole number"):
        parse_policy({"objective": "sharpe", "budget": 2.5})


def test_fingerprint_covers_the_entry_strategy_set() -> None:
    """Swapping an entry strategy must change the hash — otherwise every
    gate accepts a fit built for different strategies."""
    from midas.models import AllocationConstraints

    base = dict(
        policy=Policy(objective="sharpe"),
        tickers=["AAPL"],
        constraints=AllocationConstraints(),
        exit_params=None,
        risk_config=None,
        min_cash_pct=0.05,
        forecast_scaling="linear",
        tax_config=None,
        cash_infusion=None,
        trading_restrictions=None,
        strategy_names=["MeanReversion", "Momentum"],
    )
    ref = input_fingerprint(**base)
    assert input_fingerprint(**{**base, "strategy_names": ["Momentum", "MeanReversion"]}) == ref  # order-insensitive
    assert input_fingerprint(**{**base, "strategy_names": ["MeanReversion"]}) != ref
    assert input_fingerprint(**{**base, "strategy_names": ["MeanReversion", "RSIOversold"]}) != ref
