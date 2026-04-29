"""Regression for PR #63 bug 1: risk math must use sliced history per bar.

Two backtests cover the same prefix [start..T]. One ends at T (truncated);
the other extends past T (full). At the boundary date T, every quantity
that does NOT depend on prices[T+1:] must be identical between runs. If
any risk computation peeks into the future, the truncated terminal state
diverges from the full run's mid-window state.
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd
from conftest import make_price_frame

from midas.allocator import Allocator
from midas.backtest import BacktestEngine
from midas.models import (
    AllocationConstraints,
    Holding,
    PortfolioConfig,
    RiskConfig,
)
from midas.order_sizer import OrderSizer
from midas.strategies.mean_reversion import MeanReversion


def _build_engine(risk_config: RiskConfig) -> BacktestEngine:
    constraints = AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.95)
    allocator = Allocator(
        [(MeanReversion(window=20, threshold=0.05), 1.0)],
        constraints,
        n_tickers=2,
        risk_config=risk_config,
    )
    return BacktestEngine(
        allocator=allocator,
        order_sizer=OrderSizer(),
        exit_rules=[],
        constraints=constraints,
        train_pct=1.0,
        enable_split=False,
    )


def _run(engine: BacktestEngine, end: date) -> object:
    portfolio = PortfolioConfig(
        holdings=[
            Holding(ticker="A", shares=10, cost_basis=100.0),
            Holding(ticker="B", shares=10, cost_basis=100.0),
        ],
        available_cash=1_000.0,
    )
    price_data: dict[str, pd.DataFrame] = {
        "A": make_price_frame(start=date(2024, 1, 1), days=200, base_price=100.0, name="A"),
        "B": make_price_frame(start=date(2024, 1, 1), days=200, base_price=100.0, name="B"),
    }
    return engine.run(portfolio, price_data, start=date(2024, 1, 1), end=end)


def test_truncated_and_full_match_at_boundary() -> None:
    """A backtest ending at T and one ending later must produce identical
    state at every common bar. This pins lookahead-freedom on Phase 0
    (current_drawdown), Phase 2 (inverse-vol offset), Phase 4 (vol target),
    and the per-bar peak/equity tracking they consume.
    """
    risk = RiskConfig(
        weighting="inverse_vol",
        vol_lookback_days=30,
        vol_target=0.20,
        drawdown_penalty=1.5,
        drawdown_floor=0.5,
    )
    truncated_end = date(2024, 4, 1)
    full_end = date(2024, 6, 30)

    truncated = _run(_build_engine(risk), truncated_end)
    full = _run(_build_engine(risk), full_end)

    # Trim the full run's equity_curve to the truncated window. Each bar's
    # value must be identical — the only way the full run's bar-T value
    # could differ is if some computation consulted prices[T+1:].
    trunc_curve = {d: v for d, v in truncated.equity_curve}  # type: ignore[attr-defined]
    full_curve = {d: v for d, v in full.equity_curve}  # type: ignore[attr-defined]
    for day, value in trunc_curve.items():
        assert math.isclose(value, full_curve[day], rel_tol=1e-9), (
            f"value at {day} diverges: truncated={value} vs full={full_curve[day]}"
        )

    # Final values agree at the boundary too.
    truncated_final = truncated.final_value  # type: ignore[attr-defined]
    full_at_boundary = full_curve[truncated_end]
    assert math.isclose(truncated_final, full_at_boundary, rel_tol=1e-6)
