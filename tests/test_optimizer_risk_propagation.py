"""Regression for PR #63 bug 2: the optimizer silently dropped the risk: block.

This test feeds a non-default RiskConfig through the optimizer's
trial-construction path and asserts the configured ``vol_target`` reaches the
final BacktestResult — the only way that fact survives is if the RiskConfig
threaded through every layer (load_strategies → optimize → _init_worker →
_run_trial → Allocator → BacktestEngine → BacktestResult.risk_metrics).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from conftest import make_price_frame

from midas.models import Holding, PortfolioConfig, RiskConfig
from midas.optimizer import _run_trial


def test_risk_config_survives_run_trial() -> None:
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="A", shares=10, cost_basis=100.0)],
        available_cash=1_000.0,
    )
    price_data: dict[str, pd.DataFrame] = {
        "A": make_price_frame(start=date(2024, 1, 1), days=120, base_price=100.0, name="A"),
    }
    strategy_params = {
        "MeanReversion": {"window": 20, "threshold": 0.05},
    }
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
    # vol_target on the BacktestResult was populated from the allocator's
    # risk_config; it equals the configured value iff propagation worked.
    assert result.risk_metrics.vol_target == 0.18
