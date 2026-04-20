"""YAML config loading for portfolio and strategy definitions."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from midas.models import (
    DEFAULT_MIN_BUY_DELTA,
    DEFAULT_MIN_CASH_PCT,
    DEFAULT_SOFTMAX_TEMPERATURE,
    AllocationConstraints,
    CashInfusion,
    Holding,
    PortfolioConfig,
    StrategyConfig,
    TradingRestrictions,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        msg = f"Expected a YAML mapping in {path}"
        raise ValueError(msg)
    return data


def load_portfolio(path: Path) -> PortfolioConfig:
    """Load portfolio config from YAML."""
    raw = _load_yaml(path)

    holdings = [
        Holding(
            ticker=entry["ticker"],
            shares=float(entry["shares"]),
            cost_basis=float(entry["cost_basis"]) if "cost_basis" in entry else None,
        )
        for entry in raw["portfolio"]
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

    portfolio = PortfolioConfig(
        holdings=holdings,
        available_cash=float(raw["available_cash"]),
        cash_infusion=infusion,
        trading_restrictions=restrictions,
    )

    return portfolio


def load_strategies(
    path: Path,
) -> tuple[list[StrategyConfig], AllocationConstraints]:
    """Load strategy configs and allocation-level knobs from YAML.

    Returns (strategies, constraints) tuple. ``softmax_temperature`` and
    ``min_buy_delta`` live at the top level of the strategies file because
    they are meta-strategy knobs (how scores are blended/acted on).
    """
    raw = _load_yaml(path)

    configs = []
    for strat in raw["strategies"]:
        configs.append(
            StrategyConfig(
                name=strat["name"],
                params=strat.get("params", {}),
                tickers=strat.get("tickers"),
                weight=float(strat.get("weight", 1.0)),
            )
        )

    max_pos = raw.get("max_position_pct")
    constraints = AllocationConstraints(
        max_position_pct=float(max_pos) if max_pos is not None else None,
        min_cash_pct=float(raw.get("min_cash_pct", DEFAULT_MIN_CASH_PCT)),
        softmax_temperature=float(
            raw.get("softmax_temperature", DEFAULT_SOFTMAX_TEMPERATURE),
        ),
        min_buy_delta=float(
            raw.get("min_buy_delta", DEFAULT_MIN_BUY_DELTA),
        ),
    )
    return configs, constraints
