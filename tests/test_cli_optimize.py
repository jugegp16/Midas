"""Tests for the ``optimize`` command's objective handling."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import get_args

import pandas as pd
import pytest
import yaml
from click.testing import CliRunner
from conftest import make_price_series

import midas.cli
from midas.cli import cli
from midas.models import Objective

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


def _invoke(tmp_path: Path, portfolio_text: str, *extra: str, strategies_text: str | None = None):
    portfolio = tmp_path / "portfolio.yaml"
    portfolio.write_text(portfolio_text)
    out = tmp_path / "optimized.yaml"
    args = ["optimize", "-p", str(portfolio), "--start", "2023-01-02", "--end", "2024-06-01", "-o", str(out), "-n", "2"]
    if strategies_text is not None:
        strategies = tmp_path / "strats.yaml"
        strategies.write_text(strategies_text)
        args += ["-s", str(strategies)]
    return CliRunner().invoke(cli, [*args, *extra]), out


def test_help_shows_calmar_default_and_every_objective() -> None:
    result = CliRunner().invoke(cli, ["optimize", "--help"])
    assert "--objective" in result.output
    assert "default: calmar" in result.output
    help_text = " ".join(result.output.split())
    for name in get_args(Objective.__value__):
        assert f"{name} (" in help_text, name


def test_train_pct_leaving_no_training_days_is_a_clean_error(tmp_path: Path, fake_prices: None) -> None:
    result, _ = _invoke(tmp_path, PORTFOLIO_NO_TAX, "--train-pct", "0.001")
    assert result.exit_code == 1
    assert "leaves no training days" in result.output
    assert not isinstance(result.exception, ValueError)


def test_unknown_objective_rejected(tmp_path: Path, fake_prices: None) -> None:
    result, _ = _invoke(tmp_path, PORTFOLIO_NO_TAX, "--objective", "sortino")
    assert result.exit_code == 2
    assert "sortino" in result.output


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


def test_seed_option_is_exposed_and_accepted(tmp_path: Path, fake_prices: None) -> None:
    help_result = CliRunner().invoke(cli, ["optimize", "--help"])
    assert "--seed" in help_result.output
    assert "default: 42" in help_result.output
    result, out = _invoke(tmp_path, PORTFOLIO_NO_TAX, "--objective", "gross", "--seed", "7")
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_walk_forward_warns_when_trials_per_fold_is_too_low(tmp_path: Path, fake_prices: None) -> None:
    """The default -n over many folds lands deep in search noise; say so."""
    result, _ = _invoke(tmp_path, PORTFOLIO_NO_TAX, "--walk-forward", "-n", "20", "--objective", "gross")
    assert result.exit_code == 0, result.output
    assert "trials per fold" in result.output
    assert "noise" in result.output


def test_walk_forward_is_quiet_when_trials_per_fold_is_adequate(tmp_path: Path, fake_prices: None) -> None:
    result, _ = _invoke(tmp_path, PORTFOLIO_NO_TAX, "--walk-forward", "-n", "400", "--objective", "gross")
    assert result.exit_code == 0, result.output
    assert "trials per fold" not in result.output


STRATS_WITH_EXITS = """\
softmax_temperature: 0.4
min_buy_delta: 0.03
max_position_pct: 0.5
min_cash_pct: 0.05
strategies:
  - name: MeanReversion
    weight: 1.0
    params: {window: 20, threshold: 0.05}
  - name: StopLoss
    params: {loss_threshold: 0.08}
"""


def test_optimize_keeps_exits_and_globals_policy_owned(tmp_path: Path, fake_prices: None) -> None:
    result, out = _invoke(tmp_path, PORTFOLIO_NO_TAX, "--objective", "gross", strategies_text=STRATS_WITH_EXITS)
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(out.read_text())
    names = [s["name"] for s in data["strategies"]]
    assert "MeanReversion" in names and "StopLoss" in names
    stop = next(s for s in data["strategies"] if s["name"] == "StopLoss")
    assert stop["params"] == {"loss_threshold": 0.08}  # verbatim, not optimized
    assert data["softmax_temperature"] == 0.4  # policy-owned, round-tripped
    assert data["max_position_pct"] == 0.5


def test_search_globals_flag_restores_legacy_output(tmp_path: Path, fake_prices: None) -> None:
    result, out = _invoke(tmp_path, PORTFOLIO_NO_TAX, "--objective", "gross", "--search-globals")
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(out.read_text())
    assert "softmax_temperature" in data  # searched value emitted, as before


def test_search_globals_rejects_strategy_file_with_exits(tmp_path: Path, fake_prices: None) -> None:
    result, _ = _invoke(
        tmp_path,
        PORTFOLIO_NO_TAX,
        "--objective",
        "gross",
        "--search-globals",
        strategies_text=STRATS_WITH_EXITS,
    )
    assert result.exit_code != 0
    assert "search-globals" in result.output
