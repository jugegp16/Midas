"""Tests for the interactive tax-setup prompt (spec 2026-08-17)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from click.testing import CliRunner

import midas.cli
from midas.cli import cli
from midas.config import load_portfolio
from midas.models import Direction, HoldingPeriod, TradeRecord
from midas.trade_log import append_trade

PORTFOLIO_NO_TAX = "portfolio:\n  - ticker: AAPL\n    shares: 10\navailable_cash: 1000.0\n"


@pytest.fixture
def interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(midas.cli, "_stdin_is_interactive", lambda: True)


def _seed_log(log: Path) -> None:
    append_trade(
        log,
        TradeRecord(
            date=date(2026, 6, 1),
            ticker="AAPL",
            direction=Direction.SELL,
            shares=10.0,
            price=30.0,
            strategy_name="StopLoss",
            holding_period=HoldingPeriod.SHORT_TERM,
            purchase_date=date(2026, 1, 1),
        ),
        cost_basis=20.0,
        purchase_date=date(2026, 1, 1),
    )


def _invoke_tax_report(tmp_path: Path, portfolio: Path, user_input: str):
    log = tmp_path / "trades.csv"
    if not log.exists():
        _seed_log(log)
    return CliRunner().invoke(
        cli,
        [
            "tax-report",
            "--portfolio",
            str(portfolio),
            "--from-trades",
            str(log),
            "--year",
            "2026",
            "--output",
            str(tmp_path / "out.csv"),
        ],
        input=user_input,
    )


def test_accept_and_derive_from_income(tmp_path: Path, interactive: None) -> None:
    portfolio = tmp_path / "portfolio.yaml"
    portfolio.write_text(PORTFOLIO_NO_TAX)

    # confirm=y, mode=default (income), status=single, income, cap/lag defaults
    result = _invoke_tax_report(tmp_path, portfolio, "y\n\nsingle\n60000\n\n\n")
    assert result.exit_code == 0, result.output
    assert "Derived rates" in result.output

    reloaded = load_portfolio(portfolio)
    assert reloaded.tax_config is not None
    assert reloaded.tax_config.short_term_rate == 0.22
    assert reloaded.tax_config.long_term_rate == 0.15
    assert "2026 federal brackets" in portfolio.read_text()


def test_accept_with_direct_rates(tmp_path: Path, interactive: None) -> None:
    portfolio = tmp_path / "portfolio.yaml"
    portfolio.write_text(PORTFOLIO_NO_TAX)

    result = _invoke_tax_report(tmp_path, portfolio, "y\nrates\n0.30\n0.15\n\n\n")
    assert result.exit_code == 0, result.output

    reloaded = load_portfolio(portfolio)
    assert reloaded.tax_config is not None
    assert reloaded.tax_config.short_term_rate == 0.30
    assert reloaded.tax_config.long_term_rate == 0.15


def test_invalid_rates_reprompted(tmp_path: Path, interactive: None) -> None:
    portfolio = tmp_path / "portfolio.yaml"
    portfolio.write_text(PORTFOLIO_NO_TAX)

    # First attempt transposes the rates (lt > st) -> TaxConfig rejects -> re-asked.
    result = _invoke_tax_report(tmp_path, portfolio, "y\nrates\n0.10\n0.20\n\n\nrates\n0.30\n0.15\n\n\n")
    assert result.exit_code == 0, result.output
    assert "Invalid tax config" in result.stderr
    reloaded = load_portfolio(portfolio)
    assert reloaded.tax_config is not None
    assert reloaded.tax_config.short_term_rate == 0.30


def test_decline_writes_tax_off_and_never_asks_again(tmp_path: Path, interactive: None) -> None:
    portfolio = tmp_path / "portfolio.yaml"
    portfolio.write_text(PORTFOLIO_NO_TAX)

    first = _invoke_tax_report(tmp_path, portfolio, "n\n")
    assert first.exit_code != 0  # tax-report cannot run without rates
    assert "tax: off" in portfolio.read_text()

    second = _invoke_tax_report(tmp_path, portfolio, "")
    assert second.exit_code != 0
    assert "set up after-tax accounting" not in second.output


def test_non_interactive_never_prompts(tmp_path: Path) -> None:
    portfolio = tmp_path / "portfolio.yaml"
    portfolio.write_text(PORTFOLIO_NO_TAX)

    result = _invoke_tax_report(tmp_path, portfolio, "")
    assert result.exit_code != 0
    assert "set up after-tax accounting" not in result.output
    assert portfolio.read_text() == PORTFOLIO_NO_TAX
