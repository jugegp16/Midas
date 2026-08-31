"""Tests for the baselines artifact."""

from datetime import date
from pathlib import Path

from midas.baselines import Baselines, FoldRecord, baselines_path, read_baselines, write_baselines


def _baselines(**overrides) -> Baselines:
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
    import dataclasses

    base = Baselines(
        folds=folds,
        aggregate_oos_cagr=0.01,
        policy_hash="p" * 64,
        input_hash="i" * 64,
        validation_start=date(2016, 1, 4),
        validation_end=date(2024, 6, 28),
        cadence_days=63,
    )
    return dataclasses.replace(base, **overrides)


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


def test_includes_holdout_flag_round_trips(tmp_path: Path) -> None:
    baselines = _baselines(includes_holdout=True)
    strategies = tmp_path / "strats.yaml"
    strategies.write_text("strategies: []\n")
    write_baselines(strategies, baselines)
    assert read_baselines(strategies).includes_holdout is True


def test_includes_holdout_defaults_false_for_old_artifacts(tmp_path: Path) -> None:
    import json

    baselines = _baselines()
    strategies = tmp_path / "strats.yaml"
    strategies.write_text("strategies: []\n")
    write_baselines(strategies, baselines)
    path = baselines_path(strategies)
    raw = json.loads(path.read_text())
    raw.pop("includes_holdout", None)
    path.write_text(json.dumps(raw))
    assert read_baselines(strategies).includes_holdout is False
