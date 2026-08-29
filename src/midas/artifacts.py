"""Fit artifacts: the machine-owned side of deployment.

The strategies YAML stays human-owned; fitted members live in a sidecar
(`<stem>.members.yaml`) and every fit is appended to a history directory
(`<stem>.fits/`) so adoption is reversible. Writes are atomic (temp +
rename), mirroring the price cache.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from datetime import date
from pathlib import Path

import yaml

from midas.fitter import FitResult


def members_path(strategies_path: Path) -> Path:
    """The machine-owned members sidecar beside the strategies file."""
    return strategies_path.with_name(f"{strategies_path.stem}.members.yaml")


def fits_dir(strategies_path: Path) -> Path:
    """The fit-history directory beside the strategies file."""
    return strategies_path.with_name(f"{strategies_path.stem}.fits")


def write_fit(strategies_path: Path, fit: FitResult) -> Path:
    """Record *fit* in history and make it the deployed members.

    Returns the history entry path. A rerun of the identical fit (same
    as_of, same inputs) overwrites its own history entry rather than
    duplicating it.
    """
    history_dir = fits_dir(strategies_path)
    history_dir.mkdir(parents=True, exist_ok=True)
    entry = history_dir / f"{fit.as_of.isoformat()}-{fit.input_hash[:8]}.yaml"
    _atomic_write(entry, _serialize(fit))
    _atomic_write(members_path(strategies_path), _serialize(fit))
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
