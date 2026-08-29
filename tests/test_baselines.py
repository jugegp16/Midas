"""Tests for the baselines artifact."""

from datetime import date
from pathlib import Path

from midas.baselines import Baselines, FoldRecord, baselines_path, read_baselines, write_baselines


def _baselines() -> Baselines:
    folds = [
        FoldRecord(
            fold=1,
            test_start=date(2024, 1, 2),
            test_end=date(2024, 3, 28),
            daily_returns=[0.001, -0.002, 0.003],
            return_raw=0.002,
            return_annualized=0.0085,
            drawdown=0.002,
            after_tax_return_raw=None,
            adopted=True,
        ),
        FoldRecord(
            fold=2,
            test_start=date(2024, 3, 29),
            test_end=date(2024, 6, 28),
            daily_returns=[-0.001, 0.004],
            return_raw=0.003,
            return_annualized=0.0121,
            drawdown=0.001,
            after_tax_return_raw=0.0025,
            adopted=False,
        ),
    ]
    return Baselines(
        folds=folds,
        aggregate_oos_cagr=0.01,
        policy_hash="p" * 64,
        input_hash="i" * 64,
        validation_start=date(2016, 1, 4),
        validation_end=date(2024, 6, 28),
        cadence_days=63,
    )


def test_roundtrip(tmp_path: Path) -> None:
    strategies = tmp_path / "strategies.yaml"
    strategies.write_text("strategies: []\n")
    baselines = _baselines()
    write_baselines(strategies, baselines)
    assert read_baselines(strategies) == baselines


def test_path_location(tmp_path: Path) -> None:
    assert baselines_path(tmp_path / "strategies.yaml") == tmp_path / "strategies.baselines.json"


def test_read_none_when_absent(tmp_path: Path) -> None:
    assert read_baselines(tmp_path / "strategies.yaml") is None


def test_write_is_atomic(tmp_path: Path) -> None:
    strategies = tmp_path / "strategies.yaml"
    strategies.write_text("strategies: []\n")
    write_baselines(strategies, _baselines())
    assert not list(tmp_path.glob("*.tmp"))
