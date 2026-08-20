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


def test_load_portfolio_parses_state_file_field(tmp_path: Path) -> None:
    yaml_path = tmp_path / "portfolio.yaml"
    yaml_path.write_text(
        "state_file: ./run/portfolio.state.yaml\n"
        "portfolio:\n"
        "  - ticker: AAPL\n"
        "    shares: 100\n"
        "    cost_basis: 150\n"
        "available_cash: 1000\n"
    )
    portfolio = load_portfolio(yaml_path)
    assert portfolio.state_file == Path("./run/portfolio.state.yaml")


def test_load_portfolio_state_file_field_optional(tmp_path: Path) -> None:
    yaml_path = tmp_path / "portfolio.yaml"
    yaml_path.write_text("portfolio:\n  - ticker: AAPL\n    shares: 100\n    cost_basis: 150\navailable_cash: 1000\n")
    portfolio = load_portfolio(yaml_path)
    assert portfolio.state_file is None


def test_load_strategies(strategy_yaml: Path) -> None:
    configs, constraints, _risk = load_strategies(strategy_yaml)
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
    assert constraints.max_position_pct is None  # not specified -> None


# ---------------------------------------------------------------------------
# Risk: block parsing
# ---------------------------------------------------------------------------

import textwrap  # noqa: E402

from midas.models import RiskConfig  # noqa: E402


def _write_strategies(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "s.yaml"
    p.write_text(textwrap.dedent(body))
    return p


class TestLoadStrategiesRisk:
    def test_no_risk_block_returns_default(self, tmp_path: Path) -> None:
        path = _write_strategies(
            tmp_path,
            """
            strategies:
              - name: BollingerBand
                params: {window: 20}
            """,
        )
        _configs, _constraints, risk = load_strategies(path)
        assert risk == RiskConfig()

    def test_full_risk_block(self, tmp_path: Path) -> None:
        path = _write_strategies(
            tmp_path,
            """
            strategies:
              - name: BollingerBand
                params: {window: 20}
            risk:
              weighting: inverse_vol
              vol_lookback_days: 90
              vol_target: 0.20
              drawdown_penalty: 1.5
              drawdown_floor: 0.5
            """,
        )
        _configs, _constraints, risk = load_strategies(path)
        assert risk.weighting == "inverse_vol"
        assert risk.vol_lookback_days == 90
        assert risk.vol_target == 0.20
        assert risk.drawdown_penalty == 1.5
        assert risk.drawdown_floor == 0.5

    def test_partial_risk_block_only_vol_target(self, tmp_path: Path) -> None:
        path = _write_strategies(
            tmp_path,
            """
            strategies:
              - name: BollingerBand
                params: {window: 20}
            risk:
              vol_target: 0.18
            """,
        )
        _configs, _constraints, risk = load_strategies(path)
        assert risk.vol_target == 0.18
        assert risk.weighting == "equal"
        assert risk.drawdown_penalty is None
        assert risk.drawdown_floor is None

    def test_drawdown_one_sided_raises_at_load_time(self, tmp_path: Path) -> None:
        path = _write_strategies(
            tmp_path,
            """
            strategies:
              - name: BollingerBand
                params: {window: 20}
            risk:
              drawdown_penalty: 1.5
            """,
        )
        with pytest.raises(ValueError, match="drawdown_floor"):
            load_strategies(path)


# ---------------------------------------------------------------------------
# Tax: block parsing
# ---------------------------------------------------------------------------


def test_load_portfolio_with_tax_block(tmp_path: Path) -> None:
    """Optional tax: block parses to a TaxConfig."""
    path = tmp_path / "portfolio.yaml"
    path.write_text(
        "portfolio:\n"
        "  - ticker: AAPL\n"
        "    shares: 100\n"
        "available_cash: 1000\n"
        "tax:\n"
        "  short_term_rate: 0.32\n"
        "  long_term_rate: 0.15\n"
        "  deductible_loss_cap: 3000.0\n"
        "  payment_lag_days: 105\n"
    )
    portfolio = load_portfolio(path)
    tax = portfolio.tax_config
    assert tax is not None
    assert tax.short_term_rate == 0.32
    assert tax.long_term_rate == 0.15
    assert tax.deductible_loss_cap == 3000.0
    assert tax.payment_lag_days == 105


def test_load_portfolio_without_tax_block(tmp_path: Path) -> None:
    """Omitting tax: yields tax_config=None — after-tax accounting disabled."""
    path = tmp_path / "portfolio.yaml"
    path.write_text("portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\n")
    assert load_portfolio(path).tax_config is None


def test_load_portfolio_tax_off_declares_without_config(tmp_path: Path) -> None:
    """`tax: off` (YAML false) means: asked and declined — never prompt again."""
    path = tmp_path / "portfolio.yaml"
    path.write_text("portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\ntax: off\n")
    portfolio = load_portfolio(path)
    assert portfolio.tax_config is None
    assert portfolio.tax_declared is True


def test_load_portfolio_absent_tax_is_undeclared(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.yaml"
    path.write_text("portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\n")
    assert load_portfolio(path).tax_declared is False


def test_load_portfolio_tax_mapping_is_declared(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.yaml"
    path.write_text(
        "portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\n"
        "tax:\n  short_term_rate: 0.30\n  long_term_rate: 0.15\n"
    )
    portfolio = load_portfolio(path)
    assert portfolio.tax_declared is True
    assert portfolio.tax_config is not None


def test_load_portfolio_tax_unknown_key_raises(tmp_path: Path) -> None:
    """A typo'd rate key must not silently fall back to the default rate."""
    path = tmp_path / "portfolio.yaml"
    path.write_text(
        "portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\ntax:\n  short_termrate: 0.24\n"
    )
    with pytest.raises(ValueError, match="short_termrate"):
        load_portfolio(path)


def test_load_portfolio_tax_invalid_value_raises(tmp_path: Path) -> None:
    base = "portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\n"
    for bad in ("tax: on\n", "tax: []\n", "tax: 0\n"):
        path = tmp_path / "portfolio.yaml"
        path.write_text(base + bad)
        with pytest.raises(ValueError, match="tax:"):
            load_portfolio(path)


def test_load_portfolio_with_alerts_block(tmp_path: Path) -> None:
    """Optional alerts: block parses to an AlertsConfig."""
    path = tmp_path / "portfolio.yaml"
    path.write_text(
        "portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\n"
        "alerts:\n  discord_intra_day_channel_id: 1400000000000000001\n  timeout_seconds: 2.5\n  pending_ttl_hours: 4\n"
    )
    alerts = load_portfolio(path).alerts_config
    assert alerts is not None
    # YAML parses the snowflake as int; config normalizes to a string.
    assert alerts.discord_intra_day_channel_id == "1400000000000000001"
    assert alerts.timeout_seconds == 2.5
    assert alerts.pending_ttl_hours == 4.0


def test_load_portfolio_alerts_quoted_channel_id(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.yaml"
    path.write_text(
        "portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\n"
        'alerts:\n  discord_intra_day_channel_id: "1400000000000000001"\n'
    )
    alerts = load_portfolio(path).alerts_config
    assert alerts is not None
    assert alerts.discord_intra_day_channel_id == "1400000000000000001"


def test_load_portfolio_alerts_defaults(tmp_path: Path) -> None:
    """An empty alerts: block gets defaults (channel id may come later)."""
    path = tmp_path / "portfolio.yaml"
    path.write_text("portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\nalerts: {}\n")
    alerts = load_portfolio(path).alerts_config
    assert alerts is not None
    assert alerts.discord_intra_day_channel_id == ""
    assert alerts.timeout_seconds == 5.0
    assert alerts.pending_ttl_hours == 8.0


def test_load_portfolio_without_alerts_block(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.yaml"
    path.write_text("portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\n")
    assert load_portfolio(path).alerts_config is None


def test_load_portfolio_alerts_removed_webhook_key_raises_migration_error(tmp_path: Path) -> None:
    """The legacy webhook key must fail loudly, not be silently ignored."""
    path = tmp_path / "portfolio.yaml"
    path.write_text(
        "portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\n"
        "alerts:\n  discord_webhook_url: https://discord.com/api/webhooks/1/x\n"
    )
    with pytest.raises(ValueError, match="replaced by bot delivery"):
        load_portfolio(path)


def test_load_portfolio_alerts_unknown_key_raises(tmp_path: Path) -> None:
    """A typo'd key must not silently disable push notifications."""
    path = tmp_path / "portfolio.yaml"
    path.write_text(
        "portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\n"
        "alerts:\n  discord_channel: 1400000000000000001\n"
    )
    with pytest.raises(ValueError, match="discord_channel"):
        load_portfolio(path)


def test_load_portfolio_alerts_non_mapping_raises(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.yaml"
    path.write_text("portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\nalerts: on\n")
    with pytest.raises(ValueError, match="alerts:"):
        load_portfolio(path)


def test_load_portfolio_alerts_null_channel_parses_as_empty(tmp_path: Path) -> None:
    """A bare `discord_intra_day_channel_id:` key means "fill in later", not the string 'None'."""
    path = tmp_path / "portfolio.yaml"
    path.write_text(
        "portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\n"
        "alerts:\n  discord_intra_day_channel_id:\n"
    )
    alerts = load_portfolio(path).alerts_config
    assert alerts is not None
    assert alerts.discord_intra_day_channel_id == ""


def test_load_portfolio_alerts_non_numeric_channel_raises(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.yaml"
    path.write_text(
        "portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\n"
        "alerts:\n  discord_intra_day_channel_id: midas-alerts\n"
    )
    with pytest.raises(ValueError, match="discord_intra_day_channel_id"):
        load_portfolio(path)


def test_load_portfolio_alerts_null_timeout_raises(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.yaml"
    path.write_text(
        "portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\nalerts:\n  timeout_seconds:\n"
    )
    with pytest.raises(ValueError, match="alerts: timeout_seconds"):
        load_portfolio(path)


def test_load_portfolio_alerts_bool_timeout_raises(tmp_path: Path) -> None:
    """YAML `true` is an int subclass — float() would silently make it 1.0."""
    path = tmp_path / "portfolio.yaml"
    path.write_text(
        "portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\nalerts:\n  timeout_seconds: true\n"
    )
    with pytest.raises(ValueError, match="alerts: timeout_seconds"):
        load_portfolio(path)


def test_load_portfolio_alerts_non_numeric_ttl_raises(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.yaml"
    path.write_text(
        "portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\nalerts:\n  pending_ttl_hours: soon\n"
    )
    with pytest.raises(ValueError, match="alerts: pending_ttl_hours"):
        load_portfolio(path)


def test_load_strategies_forecast_scaling(tmp_path: Path) -> None:
    path = _write_strategies(tmp_path, "forecast_scaling: quantile\nstrategies:\n  - name: Momentum\n")
    _, constraints, _ = load_strategies(path)
    assert constraints.forecast_scaling == "quantile"


def test_load_strategies_forecast_scaling_defaults_to_none(tmp_path: Path) -> None:
    path = _write_strategies(tmp_path, "strategies:\n  - name: Momentum\n")
    _, constraints, _ = load_strategies(path)
    assert constraints.forecast_scaling == "none"


def test_load_strategies_invalid_forecast_scaling_raises(tmp_path: Path) -> None:
    path = _write_strategies(tmp_path, "forecast_scaling: softmax\nstrategies:\n  - name: Momentum\n")
    with pytest.raises(ValueError, match="forecast_scaling"):
        load_strategies(path)


def test_load_strategies_bare_forecast_scaling_key_is_default(tmp_path: Path) -> None:
    """A bare `forecast_scaling:` key means the default, not the string 'None'."""
    path = _write_strategies(tmp_path, "forecast_scaling:\nstrategies:\n  - name: Momentum\n")
    _, constraints, _ = load_strategies(path)
    assert constraints.forecast_scaling == "none"


def test_load_portfolio_alerts_report_channel(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.yaml"
    path.write_text(
        "portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\n"
        "alerts:\n  discord_intra_day_channel_id: 111\n  discord_end_of_day_channel_id: 222\n"
    )
    alerts = load_portfolio(path).alerts_config
    assert alerts is not None
    assert alerts.discord_end_of_day_channel_id == "222"


def test_load_portfolio_alerts_renamed_channel_keys_raise(tmp_path: Path) -> None:
    """The pre-rename channel keys must fail loudly with the new name."""
    base = "portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\n"
    for old_key, new_key in (
        ("discord_channel_id", "discord_intra_day_channel_id"),
        ("discord_report_channel_id", "discord_end_of_day_channel_id"),
    ):
        path = tmp_path / "portfolio.yaml"
        path.write_text(base + f"alerts:\n  {old_key}: 123\n")
        with pytest.raises(ValueError, match=new_key):
            load_portfolio(path)
