"""Tests for the ``optimize`` command's objective handling."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml
from click.testing import CliRunner
from conftest import make_price_series

import midas.cli
from midas.cli import cli

PORTFOLIO_NO_TAX = "portfolio:\n  - ticker: TEST\n    shares: 10\navailable_cash: 2000.0\n"
PORTFOLIO_TAX_OFF = PORTFOLIO_NO_TAX + "tax: off\n"
PORTFOLIO_TAXED = PORTFOLIO_NO_TAX + "tax:\n  short_term_rate: 0.35\n  long_term_rate: 0.15\n"


@pytest.fixture
def fake_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    returns = [0.005] * 100 + [-0.003] * 100 + [0.004] * 100 + [0.002] * 100
    prices = make_price_series(date(2023, 1, 2), 400, 100.0, returns, name="TEST")

    def _fetch(*_args: object, **_kwargs: object) -> dict[str, pd.DataFrame]:
        return {"TEST": prices}

    monkeypatch.setattr(midas.cli, "_fetch_prices", _fetch)
    monkeypatch.setattr(midas.cli, "_stdin_is_interactive", lambda: False)


def _invoke(tmp_path: Path, portfolio_text: str, *extra: str):
    portfolio = tmp_path / "portfolio.yaml"
    portfolio.write_text(portfolio_text)
    out = tmp_path / "optimized.yaml"
    args = ["optimize", "-p", str(portfolio), "--start", "2023-01-02", "--end", "2024-06-01", "-o", str(out), "-n", "2"]
    return CliRunner().invoke(cli, [*args, *extra]), out


def test_help_shows_sharpe_default() -> None:
    result = CliRunner().invoke(cli, ["optimize", "--help"])
    assert "--objective" in result.output
    assert "sharpe" in result.output


def test_unknown_objective_rejected(tmp_path: Path, fake_prices: None) -> None:
    result, _ = _invoke(tmp_path, PORTFOLIO_NO_TAX, "--objective", "calmar")
    assert result.exit_code == 2
    assert "calmar" in result.output


def test_net_without_tax_block_is_clean_error(tmp_path: Path, fake_prices: None) -> None:
    result, out = _invoke(tmp_path, PORTFOLIO_NO_TAX, "--objective", "net")
    assert result.exit_code != 0
    assert "--objective net" in result.output
    assert "tax:" in result.output
    assert not out.exists()


def test_net_with_tax_off_is_clean_error(tmp_path: Path, fake_prices: None) -> None:
    result, _ = _invoke(tmp_path, PORTFOLIO_TAX_OFF, "--objective", "net")
    assert result.exit_code != 0
    assert "tax:" in result.output


def test_net_with_tax_block_runs(tmp_path: Path, fake_prices: None) -> None:
    result, out = _invoke(tmp_path, PORTFOLIO_TAXED, "--objective", "net")
    assert result.exit_code == 0, result.output
    assert out.read_text().splitlines()[0] == "# optimized with --objective net"
    assert "Objective" in result.output and "net" in result.output


def test_gross_objective_writes_yaml_and_reports(tmp_path: Path, fake_prices: None) -> None:
    result, out = _invoke(tmp_path, PORTFOLIO_NO_TAX, "--objective", "gross")
    assert result.exit_code == 0, result.output
    assert out.read_text().splitlines()[0] == "# optimized with --objective gross"
    assert "strategies" in yaml.safe_load(out.read_text())


def test_walk_forward_honors_objective(tmp_path: Path, fake_prices: None) -> None:
    result, out = _invoke(tmp_path, PORTFOLIO_NO_TAX, "--walk-forward", "-n", "20", "--objective", "gross")
    assert result.exit_code == 0, result.output
    assert out.read_text().splitlines()[0] == "# optimized with --objective gross"


def test_walk_forward_report_shows_after_tax_oos_only_when_taxed(tmp_path: Path, fake_prices: None) -> None:
    taxed, _ = _invoke(tmp_path, PORTFOLIO_TAXED, "--walk-forward", "-n", "20", "--objective", "gross")
    assert taxed.exit_code == 0, taxed.output
    assert "After-Tax" in taxed.output
    untaxed, _ = _invoke(tmp_path, PORTFOLIO_NO_TAX, "--walk-forward", "-n", "20", "--objective", "gross")
    assert untaxed.exit_code == 0, untaxed.output
    assert "After-Tax" not in untaxed.output
