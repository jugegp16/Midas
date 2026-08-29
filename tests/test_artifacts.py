"""Tests for fit artifacts: members sidecar, history, rollback."""

from datetime import date
from pathlib import Path

import pytest

from midas.artifacts import fits_dir, list_fits, members_path, read_members, rollback, write_fit
from midas.fitter import FitResult


def _fit(as_of: date, score: float) -> FitResult:
    return FitResult(
        members=[{"MeanReversion": {"window": 20.0, "threshold": 0.05, "_weight": 1.0}}],
        member_scores=[score],
        restart_bests=[score],
        as_of=as_of,
        policy_hash="p" * 64,
        input_hash="i" * 64,
    )


def test_write_and_read_members_roundtrip(tmp_path: Path) -> None:
    strategies = tmp_path / "strategies.yaml"
    strategies.write_text("strategies: []\n")
    fit = _fit(date(2026, 8, 1), 1.5)
    write_fit(strategies, fit)
    assert members_path(strategies) == tmp_path / "strategies.members.yaml"
    assert read_members(strategies) == fit


def test_history_accumulates_and_rollback_restores(tmp_path: Path) -> None:
    strategies = tmp_path / "strategies.yaml"
    strategies.write_text("strategies: []\n")
    first = _fit(date(2026, 8, 1), 1.0)
    second = _fit(date(2026, 8, 2), 2.0)
    write_fit(strategies, first)
    write_fit(strategies, second)
    assert len(list_fits(strategies)) == 2
    assert read_members(strategies) == second
    restored = rollback(strategies)
    assert restored == first
    assert read_members(strategies) == first


def test_rollback_beyond_history_is_loud(tmp_path: Path) -> None:
    strategies = tmp_path / "strategies.yaml"
    strategies.write_text("strategies: []\n")
    write_fit(strategies, _fit(date(2026, 8, 1), 1.0))
    with pytest.raises(ValueError, match="history"):
        rollback(strategies, steps=3)


def test_read_members_none_when_absent(tmp_path: Path) -> None:
    strategies = tmp_path / "strategies.yaml"
    strategies.write_text("strategies: []\n")
    assert read_members(strategies) is None


def test_fits_dir_location(tmp_path: Path) -> None:
    strategies = tmp_path / "strategies.yaml"
    assert fits_dir(strategies) == tmp_path / "strategies.fits"


def test_record_fit_leaves_members_untouched(tmp_path: Path) -> None:
    from midas.artifacts import record_fit

    strategies = tmp_path / "strategies.yaml"
    strategies.write_text("strategies: []\n")
    deployed = _fit(date(2026, 8, 1), 1.0)
    write_fit(strategies, deployed)
    challenger = _fit(date(2026, 9, 1), 2.0)
    record_fit(strategies, challenger)
    assert read_members(strategies) == deployed  # not deployed
    assert len(list_fits(strategies)) == 2  # but recorded


def test_deploy_fit_writes_members_without_history(tmp_path: Path) -> None:
    from midas.artifacts import deploy_fit

    strategies = tmp_path / "strategies.yaml"
    strategies.write_text("strategies: []\n")
    fit = _fit(date(2026, 8, 1), 1.0)
    deploy_fit(strategies, fit)
    assert read_members(strategies) == fit
    assert list_fits(strategies) == []


def test_proposal_roundtrip_and_clear(tmp_path: Path) -> None:
    from datetime import datetime

    from midas.artifacts import Proposal, clear_proposal, proposal_path, read_proposal, write_proposal

    strategies = tmp_path / "strategies.yaml"
    strategies.write_text("strategies: []\n")
    created = datetime(2026, 9, 1, 20, 0, 0)
    write_proposal(
        strategies, fit_as_of=date(2026, 9, 1), input_hash="i" * 64, evidence=["drawdown breach"], created_at=created
    )
    proposal = read_proposal(strategies)
    assert proposal == Proposal(
        fit_as_of=date(2026, 9, 1),
        input_hash="i" * 64,
        evidence=["drawdown breach"],
        created_at=created,
        status="pending",
    )
    assert proposal_path(strategies).name == "strategies.proposal.yaml"
    clear_proposal(strategies)
    assert read_proposal(strategies) is None
