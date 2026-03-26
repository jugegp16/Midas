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
    }
    p = tmp_path / "portfolio.yaml"
    p.write_text(yaml.dump(data))
    return p


@pytest.fixture
def strategy_yaml(tmp_path: Path) -> Path:
    data = {
        "strategies": [
            {"name": "MeanReversion", "params": {"window": 20, "threshold": 0.08}},
            {"name": "ProfitTaking", "params": {"gain_threshold": 0.15}},
            {"name": "Momentum"},
        ],
    }
    p = tmp_path / "strategies.yaml"
    p.write_text(yaml.dump(data))
    return p


def test_load_portfolio(portfolio_yaml: Path) -> None:
    port = load_portfolio(portfolio_yaml)
    assert len(port.holdings) == 2
    assert port.holdings[0].ticker == "VOO"
    assert port.holdings[0].cost_basis == 420.0
    assert port.holdings[1].cost_basis is None
    assert port.available_cash == 2000.0
    assert port.cash_infusion is not None
    assert port.cash_infusion.amount == 1500.0
    assert port.cash_infusion.next_date == date(2026, 4, 3)
    assert port.cash_infusion.frequency == "biweekly"


def test_load_strategies(strategy_yaml: Path) -> None:
    configs = load_strategies(strategy_yaml)
    assert len(configs) == 3
    assert configs[0].name == "MeanReversion"
    assert configs[0].params["window"] == 20
    assert configs[1].name == "ProfitTaking"
    assert configs[2].name == "Momentum"
    assert configs[2].params == {}
