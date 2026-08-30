"""Tests for the pure monitor: offset ranking, hash gate, breach flag."""

from datetime import date

from midas.baselines import Baselines, FoldRecord
from midas.monitor import monitor_lines, offset_fold_rank


def _baselines(fold_returns: list[list[float]], input_hash: str = "i" * 64) -> Baselines:
    folds = []
    for i, daily in enumerate(fold_returns, start=1):
        value = 1.0
        peak = 1.0
        dd = 0.0
        for r in daily:
            value *= 1 + r
            peak = max(peak, value)
            dd = max(dd, (peak - value) / peak)
        folds.append(
            FoldRecord(
                fold=i,
                test_start=date(2024, 1, 2),
                test_end=date(2024, 3, 28),
                daily_returns=daily,
                return_raw=value - 1,
                return_annualized=value - 1,
                drawdown=dd,
                after_tax_return_raw=None,
                adopted=i == 1,
            )
        )
    return Baselines(
        folds=folds,
        aggregate_oos_cagr=0.1,
        policy_hash="p" * 64,
        input_hash=input_hash,
        validation_start=date(2016, 1, 4),
        validation_end=date(2024, 6, 28),
        cadence_days=63,
    )


BASE = _baselines([[0.001] * 63, [0.0015] * 63, [-0.0005] * 63])


def test_hash_mismatch_refuses_comparison() -> None:
    lines, breached = monitor_lines(BASE, [0.01], current_input_hash="x" * 64)
    assert len(lines) == 1 and "stale" in lines[0] and "refused" in lines[0]
    assert breached is False


def test_empty_window_is_silent() -> None:
    assert monitor_lines(BASE, [], current_input_hash="i" * 64) == ([], False)


def test_healthy_window_reports_normal() -> None:
    lines, breached = monitor_lines(BASE, [0.001] * 10, current_input_hash="i" * 64)
    assert breached is False
    assert any("day 10/63" in line for line in lines)
    assert any("pre-tax" in line for line in lines)
    assert any("normal" in line for line in lines)
    assert not any("annualized" in line.lower() for line in lines)


def test_crash_window_breaches() -> None:
    lines, breached = monitor_lines(BASE, [-0.02] * 15, current_input_hash="i" * 64)
    assert breached is True
    assert any("BREACH" in line for line in lines)


def test_offset_fold_rank_midrank_semantics() -> None:
    paths = [[0.01, 0.02], [0.01, 0.04], [0.01, 0.06]]
    assert offset_fold_rank(paths, 1, 0.01) == 0.0  # below all
    assert offset_fold_rank(paths, 1, 0.04) == 50.0  # tie counts half
    assert offset_fold_rank(paths, 1, 0.07) == 100.0  # above all
    assert offset_fold_rank(paths, 5, 0.01) is None  # no fold that long
