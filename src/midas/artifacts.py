"""Fit artifacts: the machine-owned side of deployment.

The strategies YAML stays human-owned; fitted members live in a sidecar
(`<stem>.members.yaml`) and every fit is appended to a history directory
(`<stem>.fits/`) so adoption is reversible. Writes are atomic (temp +
rename), mirroring the price cache.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import yaml

from midas.fitter import FitResult


def members_path(strategies_path: Path) -> Path:
    """The machine-owned members sidecar beside the strategies file."""
    return strategies_path.with_name(f"{strategies_path.stem}.members.yaml")


def fits_dir(strategies_path: Path) -> Path:
    """The fit-history directory beside the strategies file."""
    return strategies_path.with_name(f"{strategies_path.stem}.fits")


def record_fit(strategies_path: Path, fit: FitResult) -> Path:
    """Append *fit* to history WITHOUT deploying it.

    A rerun of the identical fit (same as_of, same inputs) overwrites its
    own history entry rather than duplicating it. Re-fit uses this: the
    fit always runs, deployment waits for adoption.
    """
    history_dir = fits_dir(strategies_path)
    history_dir.mkdir(parents=True, exist_ok=True)
    entry = history_dir / f"{fit.as_of.isoformat()}-{fit.input_hash[:8]}.yaml"
    _atomic_write(entry, _serialize(fit))
    return entry


def deploy_fit(strategies_path: Path, fit: FitResult) -> None:
    """Make *fit* the deployed members (sidecar only; history untouched)."""
    _atomic_write(members_path(strategies_path), _serialize(fit))


def write_fit(strategies_path: Path, fit: FitResult) -> Path:
    """Record *fit* in history and make it the deployed members.

    The explicit-operator path (``midas fit``): fitting by hand IS the
    adoption decision, so record and deploy compose.
    """
    entry = record_fit(strategies_path, fit)
    deploy_fit(strategies_path, fit)
    return entry


def read_members(strategies_path: Path) -> FitResult | None:
    """The currently deployed members, or None when nothing was ever fit."""
    path = members_path(strategies_path)
    if not path.exists():
        return None
    return _deserialize(path)


def list_fits(strategies_path: Path) -> list[Path]:
    """History entries, oldest first (filenames sort by as_of)."""
    history_dir = fits_dir(strategies_path)
    if not history_dir.is_dir():
        return []
    return sorted(history_dir.glob("*.yaml"))


def rollback(strategies_path: Path, steps: int = 1) -> FitResult:
    """Re-point the deployed members at the fit *steps* before the current one.

    History is never deleted — rolling back and re-adopting later both
    stay possible.

    Raises:
        ValueError: When there is no deployment or not enough history.
    """
    current = read_members(strategies_path)
    if current is None:
        msg = "nothing deployed — no members sidecar to roll back"
        raise ValueError(msg)
    entries = list_fits(strategies_path)
    current_name = f"{current.as_of.isoformat()}-{current.input_hash[:8]}.yaml"
    names = [p.name for p in entries]
    try:
        idx = names.index(current_name)
    except ValueError as exc:
        msg = f"deployed members ({current_name}) not found in fit history"
        raise ValueError(msg) from exc
    target = idx - steps
    if target < 0:
        msg = f"not enough history to roll back {steps} step(s): {idx} earlier fit(s) exist"
        raise ValueError(msg)
    restored = _deserialize(entries[target])
    _atomic_write(members_path(strategies_path), _serialize(restored))
    return restored


@dataclass(frozen=True)
class Proposal:
    """A pending adoption: a recorded fit awaiting the operator's decision."""

    fit_as_of: date
    input_hash: str
    evidence: list[str]
    created_at: datetime
    status: Literal["pending"] = "pending"


def proposal_path(strategies_path: Path) -> Path:
    """The pending-adoption proposal beside the strategies file."""
    return strategies_path.with_name(f"{strategies_path.stem}.proposal.yaml")


def write_proposal(
    strategies_path: Path,
    *,
    fit_as_of: date,
    input_hash: str,
    evidence: list[str],
    created_at: datetime,
) -> None:
    """Atomically write a pending adoption proposal."""
    payload = {
        "fit_as_of": fit_as_of.isoformat(),
        "input_hash": input_hash,
        "evidence": list(evidence),
        "created_at": created_at.isoformat(),
        "status": "pending",
    }
    _atomic_write(proposal_path(strategies_path), yaml.dump(payload, default_flow_style=False, sort_keys=False))


def read_proposal(strategies_path: Path) -> Proposal | None:
    """The pending proposal, or None."""
    path = proposal_path(strategies_path)
    if not path.exists():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Proposal(
        fit_as_of=date.fromisoformat(str(raw["fit_as_of"])),
        input_hash=str(raw["input_hash"]),
        evidence=[str(line) for line in raw.get("evidence") or []],
        created_at=datetime.fromisoformat(str(raw["created_at"])),
    )


def read_fit_entry(strategies_path: Path, fit_as_of: date, input_hash: str) -> FitResult | None:
    """Load one fit from history by its identity, or None when absent."""
    entry = fits_dir(strategies_path) / f"{fit_as_of.isoformat()}-{input_hash[:8]}.yaml"
    if not entry.exists():
        return None
    return _deserialize(entry)


def clear_proposal(strategies_path: Path) -> None:
    """Remove the pending proposal, if any."""
    proposal_path(strategies_path).unlink(missing_ok=True)


def _serialize(fit: FitResult) -> str:
    payload = asdict(fit)
    payload["as_of"] = fit.as_of.isoformat()
    return yaml.dump(payload, default_flow_style=False, sort_keys=False)


def _deserialize(path: Path) -> FitResult:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return FitResult(
        members=raw["members"],
        member_scores=[float(s) for s in raw["member_scores"]],
        restart_bests=[float(s) for s in raw["restart_bests"]],
        as_of=date.fromisoformat(str(raw["as_of"])),
        policy_hash=str(raw["policy_hash"]),
        input_hash=str(raw["input_hash"]),
    )


def _atomic_write(path: Path, content: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)
