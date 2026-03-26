"""YAML config loading for portfolio and strategy definitions."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from midas.models import CashInfusion, Holding, PortfolioConfig, StrategyConfig


def load_portfolio(path: Path) -> PortfolioConfig:
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

    return PortfolioConfig(
        holdings=holdings,
        available_cash=float(raw["available_cash"]),
        cash_infusion=infusion,
    )


def load_strategies(path: Path) -> list[StrategyConfig]:
    raw = _load_yaml(path)
    return [
        StrategyConfig(
            name=s["name"],
            params=s.get("params", {}),
            tickers=s.get("tickers"),
        )
        for s in raw["strategies"]
    ]


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        msg = f"Expected a YAML mapping in {path}"
        raise ValueError(msg)
    return data
