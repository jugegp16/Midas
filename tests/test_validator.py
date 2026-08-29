"""Tests for the policy validator and its trigger evaluation."""

from datetime import date

from midas.baselines import FoldRecord
from midas.policy import AdoptTrigger
from midas.validator import trigger_fired


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
