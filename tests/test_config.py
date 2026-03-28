"""Tests for YAML config loading."""

from datetime import date
from pathlib import Path

import pytest
import yaml

from midas.config import load_portfolio, load_strategies


@pytest.fixture
def portfolio_yaml(tmp_path: Path) -> Path:
    data = {
        "portfolio": [
            {"ticker": "VOO", "shares": 5, "cost_basis": 420.0},
            {"ticker": "AAPL", "shares": 10},
        ],
        "available_cash": 2000.0,
        "cash_infusion": {
            "amount": 1500.0,
            "next_date": "2026-04-03",
            "frequency": "biweekly",
        },
        "allocation_constraints": {
            "min_cash_pct": 0.10,
        },
    }
    p = tmp_path / "portfolio.yaml"
    p.write_text(yaml.dump(data))
    return p


@pytest.fixture
def strategy_yaml(tmp_path: Path) -> Path:
    data = {
        "sigmoid_steepness": 3.0,
        "rebalance_threshold": 0.03,
        "strategies": [
            {
                "name": "MeanReversion",
                "weight": 1.5,
                "params": {"window": 20, "threshold": 0.08},
            },
            {
                "name": "StopLoss",
                "weight": 3.0,
                "veto_threshold": -0.4,
                "params": {"loss_threshold": 0.10},
            },
            {"name": "Momentum"},
        ],
    }
    p = tmp_path / "strategies.yaml"
    p.write_text(yaml.dump(data))
    return p


def test_load_portfolio(portfolio_yaml: Path) -> None:
    port, constraints = load_portfolio(portfolio_yaml)
    assert len(port.holdings) == 2
    assert port.holdings[0].ticker == "VOO"
    assert port.holdings[0].cost_basis == 420.0
    assert port.holdings[1].cost_basis is None
    assert port.available_cash == 2000.0
    assert port.cash_infusion is not None
    assert port.cash_infusion.amount == 1500.0
    assert port.cash_infusion.next_date == date(2026, 4, 3)
    assert port.cash_infusion.frequency == "biweekly"
    # Portfolio-level constraints (position/cash limits only)
    assert constraints.min_cash_pct == 0.10
    assert constraints.max_position_pct is None  # not specified -> None


def test_load_portfolio_default_constraints(tmp_path: Path) -> None:
    data = {
        "portfolio": [{"ticker": "VOO", "shares": 5}],
        "available_cash": 1000.0,
    }
    p = tmp_path / "portfolio.yaml"
    p.write_text(yaml.dump(data))
    _port, constraints = load_portfolio(p)
    assert constraints.min_cash_pct == 0.05


def test_load_strategies(strategy_yaml: Path) -> None:
    configs, constraints = load_strategies(strategy_yaml)
    assert len(configs) == 3

    assert configs[0].name == "MeanReversion"
    assert configs[0].params["window"] == 20
    assert configs[0].weight == 1.5

    assert configs[1].name == "StopLoss"
    assert configs[1].weight == 3.0
    assert configs[1].veto_threshold == -0.4

    assert configs[2].name == "Momentum"
    assert configs[2].params == {}
    assert configs[2].weight == 1.0  # default
    assert configs[2].veto_threshold == -0.5  # default

    # Strategy-level allocation knobs
    assert constraints.sigmoid_steepness == 3.0
    assert constraints.rebalance_threshold == 0.03
