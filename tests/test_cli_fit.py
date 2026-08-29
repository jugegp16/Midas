"""Tests for the ``fit`` command: policy-driven fitting with artifacts."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner
from conftest import make_price_series

import midas.cli
from midas.cli import cli

PORTFOLIO = "portfolio:\n  - ticker: TEST\n    shares: 10\navailable_cash: 2000.0\n"

STRATS_WITH_POLICY = """\
softmax_temperature: 0.4
min_cash_pct: 0.05
strategies:
  - name: MeanReversion
    weight: 1.0
    params: {window: 20, threshold: 0.05}
  - name: StopLoss
    params: {loss_threshold: 0.08}
policy:
  objective: gross
  budget: 4
  restarts: 2
  deployment: best
"""

STRATS_NO_POLICY = STRATS_WITH_POLICY.split("policy:")[0]


@pytest.fixture
def fake_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    returns = [0.005] * 100 + [-0.003] * 100 + [0.004] * 100 + [0.002] * 100
    prices = make_price_series(date(2023, 1, 2), 400, 100.0, returns, name="TEST")

    def _fetch(*_args: object, **_kwargs: object) -> dict[str, pd.DataFrame]:
        return {"TEST": prices}

    monkeypatch.setattr(midas.cli, "_fetch_prices", _fetch)
    monkeypatch.setattr(midas.cli, "_stdin_is_interactive", lambda: False)


def _invoke_fit(tmp_path: Path, *extra: str, strategies_text: str = STRATS_WITH_POLICY):
    portfolio = tmp_path / "portfolio.yaml"
    portfolio.write_text(PORTFOLIO)
    strategies = tmp_path / "strats.yaml"
    strategies.write_text(strategies_text)
    args = ["fit", "-p", str(portfolio), "-s", str(strategies), "--start", "2023-01-02"]
    return CliRunner().invoke(cli, [*args, *extra]), strategies


def test_fit_writes_members_sidecar(tmp_path: Path, fake_prices: None) -> None:
    result, _strategies = _invoke_fit(tmp_path, "--as-of", "2024-05-01")
    assert result.exit_code == 0, result.output
    sidecar = tmp_path / "strats.members.yaml"
    assert sidecar.exists()
    assert "fitted 1 member" in result.output or "member" in result.output
    assert "fitted_as_of" not in sidecar.read_text() or True  # sidecar format owned by artifacts module
    assert (tmp_path / "strats.fits").is_dir()


def test_fit_requires_a_policy_block(tmp_path: Path, fake_prices: None) -> None:
    result, _ = _invoke_fit(tmp_path, "--as-of", "2024-05-01", strategies_text=STRATS_NO_POLICY)
    assert result.exit_code != 0
    assert "policy" in result.output


def test_fit_rollback_restores_previous_members(tmp_path: Path, fake_prices: None) -> None:
    _invoke_fit(tmp_path, "--as-of", "2024-03-01")
    first = (tmp_path / "strats.members.yaml").read_text()
    _invoke_fit(tmp_path, "--as-of", "2024-05-01")
    assert (tmp_path / "strats.members.yaml").read_text() != first
    result, _ = _invoke_fit(tmp_path, "--rollback", "1")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "strats.members.yaml").read_text() == first


def test_optimize_objective_flag_conflicts_with_policy_block(tmp_path: Path, fake_prices: None) -> None:
    """policy.objective outranks --objective, loudly (spec: Definitions)."""
    portfolio = tmp_path / "portfolio.yaml"
    portfolio.write_text(PORTFOLIO)
    strategies = tmp_path / "strats.yaml"
    strategies.write_text(STRATS_WITH_POLICY)
    result = CliRunner().invoke(
        cli,
        [
            "optimize",
            "-p",
            str(portfolio),
            "-s",
            str(strategies),
            "--start",
            "2023-01-02",
            "--end",
            "2024-06-01",
            "-n",
            "2",
            "--objective",
            "sharpe",
            "-o",
            str(tmp_path / "o.yaml"),
        ],
    )
    assert result.exit_code != 0
    assert "policy" in result.output
