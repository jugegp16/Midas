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
        "softmax_temperature": 0.25,
        "min_buy_delta": 0.03,
        "min_cash_pct": 0.10,
        "strategies": [
            {
                "name": "MeanReversion",
                "weight": 1.5,
                "params": {"window": 20, "threshold": 0.08},
            },
            {
                "name": "StopLoss",
                "params": {"loss_threshold": 0.10},
            },
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


def test_load_portfolio_minimal(tmp_path: Path) -> None:
    data = {
        "portfolio": [{"ticker": "VOO", "shares": 5}],
        "available_cash": 1000.0,
    }
    p = tmp_path / "portfolio.yaml"
    p.write_text(yaml.dump(data))
    port = load_portfolio(p)
    assert port.available_cash == 1000.0


def test_load_strategies(strategy_yaml: Path) -> None:
    configs, constraints, risk = load_strategies(strategy_yaml)
    assert len(configs) == 3

    assert configs[0].name == "MeanReversion"
    assert configs[0].params["window"] == 20
    assert configs[0].weight == 1.5

    assert configs[1].name == "StopLoss"
    assert configs[1].params["loss_threshold"] == 0.10

    assert configs[2].name == "Momentum"
    assert configs[2].params == {}
    assert configs[2].weight == 1.0  # default

    # Allocation knobs
    assert constraints.softmax_temperature == 0.25
    assert constraints.min_buy_delta == 0.03
    assert constraints.min_cash_pct == 0.10
    assert constraints.max_position_pct is None

    # Risk config defaults (no risk: block in fixture)
    assert risk.weighting == "inverse_vol"
    assert risk.vol_target_annualized == 0.20
    assert risk.idm_cap == 2.5


def test_load_strategies_parses_risk_block(tmp_path: Path) -> None:
    data = {
        "strategies": [{"name": "Momentum"}],
        "risk": {
            "weighting": "equal",
            "vol_target_annualized": 0.15,
            "idm_cap": 2.0,
            "vol_lookback_days": 30,
            "corr_lookback_days": 120,
            "vol_floor": 0.05,
        },
    }
    p = tmp_path / "strategies.yaml"
    p.write_text(yaml.dump(data))
    _, _, risk = load_strategies(p)
    assert risk.weighting == "equal"
    assert risk.vol_target_annualized == 0.15
    assert risk.idm_cap == 2.0
    assert risk.vol_lookback_days == 30
    assert risk.corr_lookback_days == 120
    assert risk.vol_floor == 0.05


def test_load_strategies_risk_validation_surfaces(tmp_path: Path) -> None:
    data = {
        "strategies": [{"name": "Momentum"}],
        "risk": {"idm_cap": 0.5},
    }
    p = tmp_path / "strategies.yaml"
    p.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="idm_cap"):
        load_strategies(p)
