"""Tests for the ``validate`` command: run the policy, write baselines."""

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
  restarts: 1
  deployment: best
  cadence_days: 40
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


def _invoke_validate(tmp_path: Path, *extra: str, strategies_text: str = STRATS_WITH_POLICY):
    portfolio = tmp_path / "portfolio.yaml"
    portfolio.write_text(PORTFOLIO)
    strategies = tmp_path / "strats.yaml"
    strategies.write_text(strategies_text)
    args = ["validate", "-p", str(portfolio), "-s", str(strategies), "--start", "2023-01-02", "--end", "2024-06-01"]
    return CliRunner().invoke(cli, [*args, *extra]), strategies


def test_validate_writes_baselines_with_matching_hash(tmp_path: Path, fake_prices: None) -> None:
    from midas.baselines import read_baselines
    from midas.config import load_policy, load_strategies
    from midas.policy import input_fingerprint

    result, strategies = _invoke_validate(tmp_path)
    assert result.exit_code == 0, result.output
    baselines = read_baselines(strategies)
    assert baselines is not None and len(baselines.folds) >= 2
    policy = load_policy(strategies)
    _configs, constraints, _risk = load_strategies(strategies)
    expected = input_fingerprint(
        policy=policy,
        tickers=["TEST"],
        constraints=constraints,
        exit_params={"StopLoss": {"loss_threshold": 0.08}},
        risk_config=_risk,
        min_cash_pct=constraints.min_cash_pct,
        forecast_scaling=constraints.forecast_scaling,
        tax_config=None,
        cash_infusion=None,
        trading_restrictions=None,
        strategy_names=["MeanReversion"],
    )
    assert baselines.input_hash == expected
    assert "fold" in result.output.lower()


def test_validate_requires_policy(tmp_path: Path, fake_prices: None) -> None:
    result, _ = _invoke_validate(tmp_path, strategies_text=STRATS_NO_POLICY)
    assert result.exit_code != 0
    assert "policy" in result.output


def test_includes_holdout_flag_stamps_the_artifact(tmp_path: Path, fake_prices: None) -> None:
    from midas.baselines import read_baselines

    result, strategies = _invoke_validate(tmp_path, "--includes-holdout")
    assert result.exit_code == 0, result.output
    assert read_baselines(strategies).includes_holdout is True
    assert "holdout" in result.output.lower()


def test_default_validate_is_pre_holdout(tmp_path: Path, fake_prices: None) -> None:
    from midas.baselines import read_baselines

    result, strategies = _invoke_validate(tmp_path)
    assert result.exit_code == 0, result.output
    assert read_baselines(strategies).includes_holdout is False


def test_validate_echoes_fold_derivation(tmp_path: Path, fake_prices: None) -> None:
    """The fold count silently depends on --start/--end — say so out loud."""
    result, _strategies = _invoke_validate(tmp_path)
    assert result.exit_code == 0, result.output
    assert "Initial train" in result.output
