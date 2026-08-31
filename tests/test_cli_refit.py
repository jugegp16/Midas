"""Tests for the ``refit`` command: record always, propose on degradation."""

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


def _setup(tmp_path: Path, *, monitor_returns: list[float] | None = None, worst_dd: float = 0.15):
    """Deployed members + baselines + a live state file with a monitor window."""
    from midas.baselines import Baselines, FoldRecord, write_baselines
    from midas.live_state import LiveState, save_atomic

    portfolio = tmp_path / "portfolio.yaml"
    portfolio.write_text(PORTFOLIO)
    strategies = tmp_path / "strats.yaml"
    strategies.write_text(STRATS_WITH_POLICY)
    # Initial deployment via midas fit.
    result = CliRunner().invoke(
        cli, ["fit", "-p", str(portfolio), "-s", str(strategies), "--start", "2023-01-02", "--as-of", "2024-01-05"]
    )
    assert result.exit_code == 0, result.output

    folds = [
        FoldRecord(
            fold=i,
            test_start=date(2023, 10, 2),
            test_end=date(2023, 12, 29),
            daily_returns=[0.001] * 60,
            return_raw=0.06,
            return_annualized=0.27,
            drawdown=worst_dd,
            after_tax_return_raw=None,
            adopted=(i == 1),
        )
        # Three folds: the drawdown arm requires >=3 priors to arm.
        for i in (1, 2, 3)
    ]
    write_baselines(
        strategies,
        Baselines(
            folds=folds,
            aggregate_oos_cagr=0.27,
            policy_hash="p" * 64,
            input_hash=_current_hash(strategies, portfolio),
            validation_start=date(2023, 1, 2),
            validation_end=date(2023, 12, 29),
            cadence_days=40,
        ),
    )

    state_path = tmp_path / "state.yaml"
    state = LiveState(available_cash=2000.0, cash_infusion_next_date=None)
    state.monitor_returns = monitor_returns or []
    save_atomic(state, state_path)
    return portfolio, strategies, state_path


def _current_hash(strategies: Path, portfolio: Path) -> str:
    from midas.config import load_policy, load_portfolio, load_strategies
    from midas.policy import input_fingerprint

    policy = load_policy(strategies)
    _configs, constraints, risk = load_strategies(strategies)
    port = load_portfolio(portfolio)
    return input_fingerprint(
        policy=policy,
        tickers=[h.ticker for h in port.holdings],
        constraints=constraints,
        exit_params={},
        risk_config=risk,
        min_cash_pct=constraints.min_cash_pct,
        forecast_scaling=constraints.forecast_scaling,
        tax_config=port.tax_config,
        cash_infusion=port.cash_infusion,
        trading_restrictions=port.trading_restrictions,
        strategy_names=["MeanReversion"],
    )


def _refit(portfolio: Path, strategies: Path, state: Path, *extra: str):
    return CliRunner().invoke(
        cli,
        [
            "refit",
            "-p",
            str(portfolio),
            "-s",
            str(strategies),
            "--start",
            "2023-01-02",
            "--as-of",
            "2024-04-01",
            "--state",
            str(state),
            *extra,
        ],
    )


def test_refit_records_without_deploying(tmp_path: Path, fake_prices: None) -> None:
    from midas.artifacts import list_fits, read_members

    portfolio, strategies, state = _setup(tmp_path)
    before = read_members(strategies)
    result = _refit(portfolio, strategies, state)
    assert result.exit_code == 0, result.output
    assert read_members(strategies) == before  # not deployed
    assert len(list_fits(strategies)) == 2  # but recorded


def test_degraded_state_writes_proposal(tmp_path: Path, fake_prices: None) -> None:
    from midas.artifacts import read_proposal

    portfolio, strategies, state = _setup(tmp_path, monitor_returns=[-0.02] * 15, worst_dd=0.01)
    result = _refit(portfolio, strategies, state)
    assert result.exit_code == 0, result.output
    proposal = read_proposal(strategies)
    assert proposal is not None
    assert proposal.fit_as_of == date(2024, 4, 1)
    assert "adopt" in result.output.lower()


def test_healthy_state_writes_no_proposal(tmp_path: Path, fake_prices: None) -> None:
    from midas.artifacts import read_proposal

    portfolio, strategies, state = _setup(tmp_path, monitor_returns=[0.001] * 15, worst_dd=0.15)
    result = _refit(portfolio, strategies, state)
    assert result.exit_code == 0, result.output
    assert read_proposal(strategies) is None


def test_adopt_deploys_latest_and_clears_proposal(tmp_path: Path, fake_prices: None) -> None:
    from midas.artifacts import read_members, read_proposal

    portfolio, strategies, state = _setup(tmp_path, monitor_returns=[-0.02] * 15, worst_dd=0.01)
    _refit(portfolio, strategies, state)
    result = _refit(portfolio, strategies, state, "--adopt")
    assert result.exit_code == 0, result.output
    assert read_members(strategies).as_of == date(2024, 4, 1)
    assert read_proposal(strategies) is None


def test_input_hash_mismatch_refuses(tmp_path: Path, fake_prices: None) -> None:
    portfolio, strategies, state = _setup(tmp_path)
    strategies.write_text(STRATS_WITH_POLICY.replace("min_cash_pct: 0.05", "min_cash_pct: 0.10"))
    result = _refit(portfolio, strategies, state)
    assert result.exit_code != 0
    assert "midas fit" in result.output


def test_stale_baselines_suppress_trigger_evaluation(tmp_path: Path, fake_prices: None) -> None:
    """Baselines validated under a different configuration prove nothing.

    The trigger compares live returns against fold distributions — if the
    baselines' input hash differs from the current configuration's, the
    comparison is meaningless and must be refused, not silently passed its
    own hash back as "current".
    """
    import dataclasses

    from midas.artifacts import read_proposal
    from midas.baselines import read_baselines, write_baselines

    portfolio, strategies, state = _setup(tmp_path, monitor_returns=[-0.02] * 15, worst_dd=0.01)
    stale = dataclasses.replace(read_baselines(strategies), input_hash="e" * 64)
    write_baselines(strategies, stale)
    result = _refit(portfolio, strategies, state)
    assert result.exit_code == 0, result.output
    assert read_proposal(strategies) is None  # degraded window, but stale evidence
    assert "different configuration" in result.output


def test_adopt_prefers_pending_proposal_over_latest_fit(tmp_path: Path, fake_prices: None) -> None:
    """--adopt confirms the standing proposal, not whatever fit came last.

    The operator is answering the proposal they were shown; a later
    trigger-silent fit recorded in between must not be deployed instead.
    """
    from midas.artifacts import read_members, read_proposal

    portfolio, strategies, state = _setup(tmp_path, monitor_returns=[-0.02] * 15, worst_dd=0.01)
    _refit(portfolio, strategies, state)  # writes proposal for fit of 2024-04-01
    assert read_proposal(strategies) is not None
    # A later trigger-silent fit lands (the monitor window recovered).
    from midas.live_state import LiveState, save_atomic

    healthy = LiveState(available_cash=2000.0, cash_infusion_next_date=None)
    healthy.monitor_returns = [0.001] * 15
    save_atomic(healthy, state)
    result = _refit(portfolio, strategies, state, "--as-of", "2024-05-01")
    assert result.exit_code == 0, result.output
    result = _refit(portfolio, strategies, state, "--adopt")
    assert result.exit_code == 0, result.output
    assert read_members(strategies).as_of == date(2024, 4, 1)  # the proposed fit, not 2024-05-01
    assert read_proposal(strategies) is None
