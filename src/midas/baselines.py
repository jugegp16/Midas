"""The baselines artifact: what validation measured, for live to compare against.

Produced by the policy validator, consumed by the monitor and the sweep.
Keyed by the full input hash so a monitor never compares live against
baselines from a different engine configuration.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class FoldRecord:
    """One validation fold's out-of-sample record, with its full daily path."""

    fold: int
    test_start: date
    test_end: date
    daily_returns: list[float]
    return_raw: float
    return_annualized: float
    drawdown: float
    after_tax_return_raw: float | None
    adopted: bool


@dataclass(frozen=True)
class Baselines:
    """Validation output for one (policy, portfolio, configuration) triple."""

    folds: list[FoldRecord]
    aggregate_oos_cagr: float
    policy_hash: str
    input_hash: str
    validation_start: date
    validation_end: date
    cadence_days: int


def baselines_path(strategies_path: Path) -> Path:
    """The baselines artifact beside the strategies file."""
    return strategies_path.with_name(f"{strategies_path.stem}.baselines.json")


def write_baselines(strategies_path: Path, baselines: Baselines) -> Path:
    """Atomically write the baselines artifact. Returns its path."""
    path = baselines_path(strategies_path)
    payload = asdict(baselines)
    for fold in payload["folds"]:
        fold["test_start"] = fold["test_start"].isoformat()
        fold["test_end"] = fold["test_end"].isoformat()
    payload["validation_start"] = baselines.validation_start.isoformat()
    payload["validation_end"] = baselines.validation_end.isoformat()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp_path, path)
    return path


def read_baselines(strategies_path: Path) -> Baselines | None:
    """Read the baselines artifact, or None when absent."""
    path = baselines_path(strategies_path)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    folds = [
        FoldRecord(
            fold=int(f["fold"]),
            test_start=date.fromisoformat(f["test_start"]),
            test_end=date.fromisoformat(f["test_end"]),
            daily_returns=[float(r) for r in f["daily_returns"]],
            return_raw=float(f["return_raw"]),
            return_annualized=float(f["return_annualized"]),
            drawdown=float(f["drawdown"]),
            after_tax_return_raw=(None if f["after_tax_return_raw"] is None else float(f["after_tax_return_raw"])),
            adopted=bool(f["adopted"]),
        )
        for f in raw["folds"]
    ]
    return Baselines(
        folds=folds,
        aggregate_oos_cagr=float(raw["aggregate_oos_cagr"]),
        policy_hash=str(raw["policy_hash"]),
        input_hash=str(raw["input_hash"]),
        validation_start=date.fromisoformat(raw["validation_start"]),
        validation_end=date.fromisoformat(raw["validation_end"]),
        cadence_days=int(raw["cadence_days"]),
    )
