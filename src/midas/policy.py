"""The deployment policy: what gets fit, how, and how it is deployed.

A policy is the unit that gets validated, deployed, and monitored. The
walk-forward validator executes it per fold and live's scheduled re-fit
executes it going forward — one procedure, one code path. This module is
pure: parsing, hashing, and seed derivation only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Literal, get_args

from midas.models import (
    DEFAULT_OBJECTIVE,
    AllocationConstraints,
    Objective,
    RiskConfig,
    TaxConfig,
    objective_error,
)

type Deployment = Literal["best", "ensemble"]


@dataclass(frozen=True)
class AdoptTrigger:
    """When live proposes adopting a challenger fit.

    Adoption is degradation-gated, never score-gated: the trailing
    fold-window must look bad by evidence the search was not selected on.
    """

    oos_percentile_below: float = 10.0
    drawdown_breach: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.oos_percentile_below <= 100:
            msg = f"oos_percentile_below must be in [0, 100], got {self.oos_percentile_below}"
            raise ValueError(msg)


@dataclass(frozen=True)
class Policy:
    """A deployment policy: fit procedure, selection rule, and cadence."""

    objective: Objective = DEFAULT_OBJECTIVE
    budget: int = 2000
    restarts: int = 4
    base_seed: int = 42
    deployment: Deployment = "best"
    ensemble_size: int = 20
    cadence_days: int = 63
    adopt_trigger: AdoptTrigger = field(default_factory=AdoptTrigger)

    def __post_init__(self) -> None:
        if self.objective not in get_args(Objective.__value__):
            raise ValueError(objective_error(self.objective))
        if self.deployment not in get_args(Deployment.__value__):
            msg = f"deployment must be 'best' or 'ensemble', got {self.deployment!r}"
            raise ValueError(msg)
        for name in ("budget", "restarts", "ensemble_size", "cadence_days"):
            if getattr(self, name) < 1:
                msg = f"{name} must be >= 1, got {getattr(self, name)}"
                raise ValueError(msg)
        if self.deployment == "ensemble" and self.ensemble_size < self.restarts:
            msg = (
                f"ensemble_size ({self.ensemble_size}) must be >= restarts ({self.restarts}) "
                "so every restart stratum contributes at least one member"
            )
            raise ValueError(msg)


def parse_policy(raw: Mapping[str, object]) -> Policy:
    """Build a Policy from a YAML ``policy:`` block, loudly rejecting unknowns.

    Raises:
        ValueError: On unknown keys or invalid field values.
    """
    known = {f.name for f in dataclasses.fields(Policy)}
    unknown = set(raw) - known
    if unknown:
        msg = f"policy: has unrecognized keys {sorted(unknown)}; known keys: {sorted(known)}"
        raise ValueError(msg)
    kwargs: dict[str, object] = dict(raw)
    trigger_raw = kwargs.pop("adopt_trigger", None)
    if trigger_raw is not None:
        if not isinstance(trigger_raw, Mapping):
            msg = f"adopt_trigger must be a mapping, got {type(trigger_raw).__name__}"
            raise ValueError(msg)
        trigger_known = {f.name for f in dataclasses.fields(AdoptTrigger)}
        trigger_unknown = set(trigger_raw) - trigger_known
        if trigger_unknown:
            msg = f"adopt_trigger has unrecognized keys {sorted(trigger_unknown)}; known: {sorted(trigger_known)}"
            raise ValueError(msg)
        kwargs["adopt_trigger"] = AdoptTrigger(
            oos_percentile_below=float(trigger_raw.get("oos_percentile_below", 10.0)),
            drawdown_breach=bool(trigger_raw.get("drawdown_breach", True)),
        )
    for int_field in ("budget", "restarts", "base_seed", "ensemble_size", "cadence_days"):
        if int_field in kwargs:
            value = kwargs[int_field]
            if not isinstance(value, int | float | str):
                msg = f"policy.{int_field} must be a number, got {type(value).__name__}"
                raise ValueError(msg)
            kwargs[int_field] = int(value)
    return Policy(**kwargs)  # type: ignore[arg-type]


def policy_hash(policy: Policy) -> str:
    """SHA-256 of the behavior-affecting policy fields under the active mode.

    ``ensemble_size`` is excluded under ``deployment: best`` so editing an
    inert field never invalidates artifacts.
    """
    payload = dataclasses.asdict(policy)
    if policy.deployment == "best":
        payload.pop("ensemble_size")
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def derive_seed(base_seed: int, as_of: date, restart: int) -> int:
    """Deterministic sampler seed for one restart of one fit.

    Derived from the fit date rather than a fold index so the validator's
    per-fold fits and a standalone fit at the same date draw identical
    seeds — the coherence principle applied to the RNG. Stable across
    processes and machines (unlike ``hash()``).
    """
    digest = hashlib.sha256(f"{base_seed}:{as_of.isoformat()}:{restart}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def input_fingerprint(
    *,
    policy: Policy,
    tickers: Sequence[str],
    constraints: AllocationConstraints,
    exit_params: Mapping[str, Mapping[str, float | int | str]] | None,
    risk_config: RiskConfig | None,
    min_cash_pct: float,
    forecast_scaling: str,
    tax_config: TaxConfig | None,
) -> str:
    """SHA-256 over everything a fit's outcome depends on besides prices.

    The monitor and re-fit refuse to act on artifacts whose fingerprint
    does not match the running configuration — comparing against a
    different engine silently is the failure this exists to prevent.
    """
    payload = {
        "policy": _mode_aware_policy_payload(policy),
        "tickers": sorted(tickers),
        "constraints": dataclasses.asdict(constraints),
        "exit_params": {k: dict(v) for k, v in (exit_params or {}).items()},
        "risk": dataclasses.asdict(risk_config) if risk_config is not None else None,
        "min_cash_pct": min_cash_pct,
        "forecast_scaling": forecast_scaling,
        "tax": dataclasses.asdict(tax_config) if tax_config is not None else None,
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _mode_aware_policy_payload(policy: Policy) -> dict[str, object]:
    payload = dataclasses.asdict(policy)
    if policy.deployment == "best":
        payload.pop("ensemble_size")
    return payload
