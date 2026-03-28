"""YAML config loading for portfolio and strategy definitions."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from midas.models import (
    AllocationConstraints,
    CashInfusion,
    Holding,
    PortfolioConfig,
    StrategyConfig,
    TradingRestrictions,
)


def load_portfolio(path: Path) -> tuple[PortfolioConfig, AllocationConstraints]:
    """Load portfolio config and allocation constraints from YAML.

    Returns (portfolio, constraints) tuple.
    """
    raw = _load_yaml(path)

    holdings = [
        Holding(
            ticker=h["ticker"],
            shares=float(h["shares"]),
            cost_basis=float(h["cost_basis"]) if "cost_basis" in h else None,
        )
        for h in raw["portfolio"]
    ]

    infusion = None
    if "cash_infusion" in raw:
        ci = raw["cash_infusion"]
        next_date = ci["next_date"]
        if isinstance(next_date, str):
            next_date = date.fromisoformat(next_date)
        elif isinstance(next_date, datetime):
            next_date = next_date.date()
        infusion = CashInfusion(
            amount=float(ci["amount"]),
            next_date=next_date,
            frequency=ci.get("frequency"),
        )

    restrictions = None
    if "trading_restrictions" in raw:
        tr = raw["trading_restrictions"]
        restrictions = TradingRestrictions(
            round_trip_days=int(tr.get("round_trip_days", 0)),
        )

    # Parse allocation constraints
    constraints = AllocationConstraints()
    if "allocation_constraints" in raw:
        ac = raw["allocation_constraints"]
        constraints = AllocationConstraints(
            max_position_pct=ac.get("max_position_pct"),
            min_cash_pct=float(ac.get("min_cash_pct", 0.05)),
            rebalance_threshold=float(ac.get("rebalance_threshold", 0.02)),
            sigmoid_steepness=float(ac.get("sigmoid_steepness", 2.0)),
        )

    portfolio = PortfolioConfig(
        holdings=holdings,
        available_cash=float(raw["available_cash"]),
        cash_infusion=infusion,
        trading_restrictions=restrictions,
    )

    return portfolio, constraints


def load_strategies(path: Path) -> list[StrategyConfig]:
    raw = _load_yaml(path)
    configs = []
    for s in raw["strategies"]:
        configs.append(StrategyConfig(
            name=s["name"],
            params=s.get("params", {}),
            tickers=s.get("tickers"),
            weight=float(s.get("weight", 1.0)),
            veto_threshold=float(s.get("veto_threshold", -0.5)),
        ))
    return configs


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        msg = f"Expected a YAML mapping in {path}"
        raise ValueError(msg)
    return data
