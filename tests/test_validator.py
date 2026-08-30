"""Tests for the policy validator and its trigger evaluation."""

from datetime import date

import numpy as np
import pandas as pd
from conftest import make_price_series

from midas.baselines import FoldRecord
from midas.models import AllocationConstraints, Holding, PortfolioConfig
from midas.policy import AdoptTrigger, Policy
from midas.validator import trigger_fired, validate_policy


def _fold(fold: int, daily: list[float]) -> FoldRecord:
    value = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in daily:
        value *= 1 + r
        peak = max(peak, value)
        max_dd = max(max_dd, (peak - value) / peak)
    return FoldRecord(
        fold=fold,
        test_start=date(2024, 1, 2),
        test_end=date(2024, 3, 28),
        daily_returns=daily,
        return_raw=value - 1,
        return_annualized=value - 1,
        drawdown=max_dd,
        after_tax_return_raw=None,
        adopted=False,
    )


HEALTHY = [0.001] * 60
CRASH = [-0.01] * 60
TRIGGER = AdoptTrigger(oos_percentile_below=10.0, drawdown_breach=True)


def test_no_priors_never_fires() -> None:
    assert trigger_fired([], CRASH, TRIGGER) is False


def test_drawdown_breach_fires_with_one_prior() -> None:
    priors = [_fold(1, HEALTHY)]  # tiny drawdown history
    assert trigger_fired(priors, CRASH, TRIGGER) is True


def test_drawdown_arm_can_be_disabled() -> None:
    priors = [_fold(1, HEALTHY)]
    off = AdoptTrigger(oos_percentile_below=10.0, drawdown_breach=False)
    # 4 priors < percentile arming threshold, drawdown arm off -> silent.
    assert trigger_fired(priors, CRASH, off) is False


def test_percentile_arm_silent_below_five_priors_then_fires() -> None:
    # Priors with real drawdowns so the crash path does NOT breach worst-DD:
    # each prior dips 40% mid-fold then recovers; the crash path's DD (~45%)
    # stays under some priors' but its RETURN is bottom-decile.
    dippy = [-0.02] * 25 + [0.03] * 35  # deep dip, strong recovery, positive return
    off_dd = AdoptTrigger(oos_percentile_below=10.0, drawdown_breach=False)
    four = [_fold(i, dippy) for i in range(4)]
    assert trigger_fired(four, CRASH, off_dd) is False  # arm not yet armed
    five = [_fold(i, dippy) for i in range(5)]
    assert trigger_fired(five, CRASH, off_dd) is True  # bottom-decile path


def test_healthy_path_with_many_priors_is_silent() -> None:
    priors = [_fold(i, [0.001 * (1 + i % 3)] * 60) for i in range(6)]
    assert trigger_fired(priors, HEALTHY, TRIGGER) is False


# ---------------------------------------------------------------------------
# validate_policy
# ---------------------------------------------------------------------------


def _val_data() -> tuple[PortfolioConfig, dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(5)
    prices = make_price_series(date(2023, 1, 2), 240, 100.0, list(rng.normal(0.0005, 0.012, 240)), name="TEST")
    portfolio = PortfolioConfig(holdings=[Holding(ticker="TEST", shares=10, cost_basis=100.0)], available_cash=2000.0)
    return portfolio, {"TEST": prices}


TINY = Policy(objective="gross", budget=4, restarts=1, base_seed=42, deployment="best", cadence_days=40)


def _validate(portfolio, price_data, policy=TINY):
    days = sorted(price_data["TEST"].index)
    return validate_policy(
        portfolio,
        price_data,
        days[0],
        days[-1],
        policy,
        constraints=AllocationConstraints(),
        exit_params={},
        strategy_names=["MeanReversion"],
    )


def test_validator_produces_baselines() -> None:
    portfolio, price_data = _val_data()
    baselines = _validate(portfolio, price_data)
    assert len(baselines.folds) >= 2
    assert all(fold.daily_returns for fold in baselines.folds)
    assert baselines.folds[0].adopted is True
    assert baselines.cadence_days == 40
    assert len(baselines.policy_hash) == 64 and len(baselines.input_hash) == 64
    assert np.isfinite(baselines.aggregate_oos_cagr)


def test_adoption_chain_keeps_incumbent_when_trigger_silent(monkeypatch) -> None:
    import midas.validator as validator_module

    monkeypatch.setattr(validator_module, "trigger_fired", lambda *a, **k: False)
    portfolio, price_data = _val_data()
    baselines = _validate(portfolio, price_data)
    assert baselines.folds[0].adopted is True
    assert all(fold.adopted is False for fold in baselines.folds[1:])


def test_adoption_chain_adopts_on_trigger(monkeypatch) -> None:
    import midas.validator as validator_module

    monkeypatch.setattr(validator_module, "trigger_fired", lambda *a, **k: True)
    portfolio, price_data = _val_data()
    baselines = _validate(portfolio, price_data)
    assert all(fold.adopted is True for fold in baselines.folds)


def test_validator_is_deterministic() -> None:
    portfolio, price_data = _val_data()
    a = _validate(portfolio, price_data)
    b = _validate(portfolio, price_data)
    assert a == b


def test_folds_carry_the_portfolio(monkeypatch) -> None:
    import midas.validator as validator_module

    calls: list[tuple[date, bool]] = []
    real_run = validator_module.BacktestEngine.run

    def spy(self, portfolio, price_data, start, end, carried=None):
        calls.append((start, carried is not None))
        return real_run(self, portfolio, price_data, start, end, carried=carried)

    monkeypatch.setattr(validator_module.BacktestEngine, "run", spy)
    portfolio, price_data = _val_data()
    _validate(portfolio, price_data)
    data_start = sorted(price_data["TEST"].index)[0]
    # The class-level spy also sees the fitter's internal full-history runs;
    # fold runs are the ones that do not start at the data start.
    fold_runs = [carried for start, carried in calls if start != data_start]
    assert fold_runs, "no fold runs observed"
    assert fold_runs[0] is False  # first fold starts the book
    assert all(fold_runs[1:])  # every later fold resumes it


def test_disabled_trigger_never_fires() -> None:
    """The frozen policy holds through any crash — that is its definition."""
    from midas.policy import AdoptTrigger

    priors = [_fold(i, [0.001] * 60) for i in range(6)]
    crash = [-0.05] * 30
    assert trigger_fired(priors, crash, AdoptTrigger.never()) is False


def test_percentile_zero_disables_that_arm() -> None:
    """oos_percentile_below: 0 means "no percentile arm", not "below the min"."""
    from midas.policy import AdoptTrigger

    priors = [_fold(i, [0.001] * 60) for i in range(6)]
    below_min = [-0.001] * 60  # under every prior path at every offset
    trigger = AdoptTrigger(oos_percentile_below=0.0, drawdown_breach=False)
    assert trigger_fired(priors, below_min, trigger) is False


def test_trigger_and_monitor_agree_on_the_rank() -> None:
    """The trigger fires on the same number the operator reads.

    Six prior folds, one deeply negative; the window sits between the
    worst fold and the rest — midrank p17. The monitor prints p17, a 10%%
    trigger stays silent, a 20%% trigger fires: one number governs both.
    """
    from datetime import date as date_type

    from midas.baselines import Baselines
    from midas.monitor import monitor_lines

    priors = [_fold(i, [0.002] * 60) for i in range(5)] + [_fold(5, [-0.004] * 60)]
    window = [-0.002] * 40  # cumulative -7.7%: above the worst fold's -14.5%
    assert trigger_fired(priors, window, AdoptTrigger(oos_percentile_below=10.0, drawdown_breach=False)) is False
    assert trigger_fired(priors, window, AdoptTrigger(oos_percentile_below=20.0, drawdown_breach=False)) is True
    baselines = Baselines(
        folds=priors,
        aggregate_oos_cagr=0.1,
        policy_hash="p" * 64,
        input_hash="i" * 64,
        validation_start=date_type(2024, 1, 2),
        validation_end=date_type(2025, 6, 30),
        cadence_days=63,
    )
    lines, _breached = monitor_lines(baselines, window, current_input_hash="i" * 64)
    assert any("p17 of validation folds" in line for line in lines), lines


def test_frozen_policy_fits_only_fold_zero(monkeypatch) -> None:
    """With the trigger disabled a challenger can never be adopted — the
    validator must not spend a full search budget producing a discarded
    fit every fold (it doubles the cost of the hold-vs-refit experiment)."""
    import midas.validator as validator_module

    calls: list[object] = []
    real_fit = validator_module.fit_as_of

    def counting_fit(*args, **kwargs):
        calls.append(1)
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(validator_module, "fit_as_of", counting_fit)
    import dataclasses

    portfolio, price_data = _val_data()
    frozen = dataclasses.replace(TINY, adopt_trigger=AdoptTrigger.never())
    baselines = _validate(portfolio, price_data, frozen)
    assert len(baselines.folds) >= 2
    assert len(calls) == 1  # fold 0 only
    assert all(fold.adopted == (fold.fold == baselines.folds[0].fold) for fold in baselines.folds)
