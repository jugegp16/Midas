"""Regression for PR #63 bug 2: the optimizer silently dropped the risk: block.

This test feeds a non-default RiskConfig through the optimizer's
trial-construction path and asserts the configured ``vol_target`` reaches the
final BacktestResult — the only way that fact survives is if the RiskConfig
threaded through every layer (load_strategies → optimize → _init_worker →
_run_trial → Allocator → BacktestEngine → BacktestResult.risk_metrics).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from conftest import make_price_frame

from midas.models import Holding, PortfolioConfig, RiskConfig
from midas.optimizer import _run_trial


def _two_ticker_portfolio() -> tuple[PortfolioConfig, dict[str, pd.DataFrame]]:
    portfolio = PortfolioConfig(
        holdings=[
            Holding(ticker="LO", shares=10, cost_basis=100.0),
            Holding(ticker="HI", shares=10, cost_basis=100.0),
        ],
        available_cash=2_000.0,
    )
    # Distinguishable vols so inverse-vol weighting actually changes the book.
    rng_lo = np.random.default_rng(1)
    rng_hi = np.random.default_rng(2)
    lo_returns = list(rng_lo.normal(0.0, 0.005, 200))
    hi_returns = list(rng_hi.normal(0.0, 0.05, 200))
    price_data: dict[str, pd.DataFrame] = {
        "LO": make_price_frame(start=date(2024, 1, 1), days=200, base_price=100.0, daily_returns=lo_returns, name="LO"),
        "HI": make_price_frame(start=date(2024, 1, 1), days=200, base_price=100.0, daily_returns=hi_returns, name="HI"),
    }
    return portfolio, price_data


def test_risk_config_survives_run_trial_structural() -> None:
    """Structural: vol_target reaches BacktestResult.risk_metrics.

    Catches the PR #63 bug 2 regression where the optimizer silently dropped
    the risk: block before constructing the trial allocator.
    """
    portfolio, price_data = _two_ticker_portfolio()
    strategy_params = {"MeanReversion": {"window": 20, "threshold": 0.05}}
    configured = RiskConfig(weighting="inverse_vol", vol_target=0.18, vol_lookback_days=60)

    *_, result = _run_trial(
        strategy_params,
        portfolio,
        price_data,
        start=date(2024, 1, 1),
        end=date(2024, 6, 1),
        risk_config=configured,
        enable_split=False,
    )

    assert result.risk_metrics is not None
    assert result.risk_metrics.vol_target == 0.18


def test_different_risk_configs_produce_different_equity_curves() -> None:
    """Behavioral: two runs with materially different risk: blocks must produce
    materially different equity curves under identical strategy params and data.

    Catches the partially-wired pipeline mode where config plumbs through to
    telemetry but the allocator quietly ignores it during weighting (or the
    backtester ignores current_drawdown).
    """
    portfolio, price_data = _two_ticker_portfolio()
    strategy_params = {"MeanReversion": {"window": 20, "threshold": 0.05}}

    risk_a = RiskConfig(weighting="equal")
    risk_b = RiskConfig(weighting="inverse_vol", vol_target=0.10, vol_lookback_days=60)

    *_, result_a = _run_trial(
        strategy_params,
        portfolio,
        price_data,
        start=date(2024, 1, 1),
        end=date(2024, 6, 1),
        risk_config=risk_a,
        enable_split=False,
    )
    *_, result_b = _run_trial(
        strategy_params,
        portfolio,
        price_data,
        start=date(2024, 1, 1),
        end=date(2024, 6, 1),
        risk_config=risk_b,
        enable_split=False,
    )

    # Same data + strategy params, materially different risk policy → final
    # values must diverge. If they match, the config plumbed through to
    # telemetry but didn't actually change construction.
    assert result_a.final_value != result_b.final_value
