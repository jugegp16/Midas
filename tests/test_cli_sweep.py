"""Tests for the ``sweep`` command."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner
from conftest import make_price_series

import midas.cli
from midas.cli import cli
from midas.sweep import SweepReport

PORTFOLIO = "portfolio:\n  - ticker: TEST\n    shares: 10\navailable_cash: 2000.0\n"

STRATS_WITH_POLICY = """\
min_cash_pct: 0.05
strategies:
  - name: MeanReversion
    weight: 1.0
    params: {window: 20, threshold: 0.05}
policy:
  objective: gross
  budget: 4
  restarts: 1
  deployment: best
  cadence_days: 40
"""


@pytest.fixture
def fake_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    returns = [0.005] * 100 + [-0.003] * 100 + [0.004] * 100 + [0.002] * 100
    prices = make_price_series(date(2023, 1, 2), 400, 100.0, returns, name="TEST")

    def _fetch(*_args: object, **_kwargs: object) -> dict[str, pd.DataFrame]:
        return {"TEST": prices}

    monkeypatch.setattr(midas.cli, "_fetch_prices", _fetch)
    monkeypatch.setattr(midas.cli, "_stdin_is_interactive", lambda: False)


def _invoke_sweep(tmp_path: Path, *extra: str):
    portfolio = tmp_path / "portfolio.yaml"
    portfolio.write_text(PORTFOLIO)
    strategies = tmp_path / "strats.yaml"
    strategies.write_text(STRATS_WITH_POLICY)
    args = ["sweep", "-p", str(portfolio), "-s", str(strategies), "--start", "2023-01-02", "--end", "2024-06-01"]
    return CliRunner().invoke(cli, [*args, *extra])


def test_sweep_trims_holdout_and_preflights(tmp_path: Path, fake_prices: None, monkeypatch) -> None:
    seen: dict = {}

    def recorder(*args, **kwargs):
        seen["end"] = args[3]
        seen["holdout"] = kwargs.get("holdout_trimmed_to")
        return SweepReport(cells=[])

    monkeypatch.setattr(midas.cli, "run_sweep", recorder)
    result = _invoke_sweep(tmp_path, "--holdout-days", "100", "--seeds", "1")
    assert result.exit_code == 0, result.output
    assert seen["end"] == date(2024, 6, 1) - pd.Timedelta(days=100).to_pytimedelta()
    assert "holdout" in result.output.lower()
    assert "trials" in result.output.lower()  # preflight


def test_use_holdout_requires_single_variant(tmp_path: Path, fake_prices: None) -> None:
    result = _invoke_sweep(tmp_path, "--use-holdout", "--vary", "objective", "--value", "gross", "--value", "calmar")
    assert result.exit_code != 0
    assert "holdout" in result.output.lower()


def test_budget_override_needs_force(tmp_path: Path, fake_prices: None) -> None:
    result = _invoke_sweep(tmp_path, "--budget", "2")
    assert result.exit_code != 0
    assert "force" in result.output.lower()


def test_sweep_runs_end_to_end_tiny(tmp_path: Path, fake_prices: None) -> None:
    import json

    result = _invoke_sweep(tmp_path, "--seeds", "1", "--holdout-days", "30")
    assert result.exit_code == 0, result.output
    assert "mean OOS CAGR" in result.output
    checkpoints = list(tmp_path.glob("*.sweep-cells.jsonl"))
    assert checkpoints, "per-cell checkpoint file missing"
    lines = [json.loads(x) for x in checkpoints[0].read_text().splitlines()]
    assert lines[0].get("run_header") is True  # runs are delimited in the append-only file
    assert any("aggregate_oos_cagr" in x for x in lines[1:])
