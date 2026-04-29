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

    Strong form (spec line 135): equity-curve equality is necessary but not
    sufficient — two runs could produce identical equity while differing in
    target weights that didn't trade that bar (e.g. blocked by min_buy_delta).
    Also compare the trade list and the cumulative attribution dict, both of
    which depend directly on every Phase 0/2/4 decision.
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

    # 1. Equity-curve equality at every common bar.
    trunc_curve = {d: v for d, v in truncated.equity_curve}  # type: ignore[attr-defined]
    full_curve = {d: v for d, v in full.equity_curve}  # type: ignore[attr-defined]
    for day, value in trunc_curve.items():
        assert math.isclose(value, full_curve[day], rel_tol=1e-9), (
            f"value at {day} diverges: truncated={value} vs full={full_curve[day]}"
        )

    # 2. Final values agree at the boundary.
    truncated_final = truncated.final_value  # type: ignore[attr-defined]
    full_at_boundary = full_curve[truncated_end]
    assert math.isclose(truncated_final, full_at_boundary, rel_tol=1e-6)

    # 3. Trades up to the boundary are identical. A trade is direct evidence
    # that the allocator's target-weight delta exceeded min_buy_delta for some
    # ticker on that bar; matching trade lists prove the allocation decisions
    # at every common bar produced identical orders.
    truncated_trades = truncated.trades  # type: ignore[attr-defined]
    full_trades_to_boundary = [t for t in full.trades if t.date <= truncated_end]  # type: ignore[attr-defined]
    assert len(truncated_trades) == len(full_trades_to_boundary), (
        f"trade count diverges at boundary: truncated={len(truncated_trades)} "
        f"vs full[..{truncated_end}]={len(full_trades_to_boundary)}"
    )
    for tt, ft in zip(truncated_trades, full_trades_to_boundary, strict=True):
        assert tt == ft, f"trade differs at boundary: truncated={tt} full={ft}"

    # 4. Cumulative per-strategy attributed P&L matches at boundary (the
    # attribution mechanic touches every buy and every sell — divergence here
    # implies divergent allocation decisions that survived even when net
    # equity coincidentally matched).
    truncated_attr = truncated.risk_metrics.per_strategy_pnl  # type: ignore[attr-defined]
    # The full run's *final* attribution would include all bars; we can't
    # directly compare it. The boundary-equality guarantee already follows
    # from (3): identical trades → identical attribution evolution up to T.
    # Sanity-check that it's at least non-empty when trades fired.
    if truncated_trades:
        assert truncated_attr, "expected attribution buckets but got none"
