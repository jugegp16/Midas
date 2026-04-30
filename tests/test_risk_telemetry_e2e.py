"""End-to-end regression for risk telemetry plumbing through the backtest engine.

These tests run a full ``BacktestEngine.run`` and assert that the resulting
``RiskMetrics`` reflects what the risk engine actually did. The bug they
guard against (PR #65 critical issue): the engine reconstructs
``AllocationResult`` after exit-rule clamping and target rebalancing, and
historically dropped ``risk_telemetry`` on those copies. With telemetry
dropped, every per-bar metric (CPPI scale, vol-target scale, gross exposure)
defaults to its inert value and the headline observability deliverable is
silently broken.

Allocator-level tests can't catch this — they bypass the engine.
"""

from __future__ import annotations

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


def _run_backtest(risk: RiskConfig) -> object:
    """Run a synthetic 200-bar backtest with a deep drawdown mid-window.

    Returns and price drops are tuned to produce a >40% drawdown by day ~60,
    enough to make CPPI bind hard if it's wired correctly. With
    ``drawdown_penalty=2.0`` and ``drawdown_floor=0.4``, exposure should
    scale to the floor and stay there until the portfolio recovers.
    """
    constraints = AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.95)
    allocator = Allocator(
        [(MeanReversion(window=20, threshold=0.05), 1.0)],
        constraints,
        n_tickers=2,
        risk_config=risk,
    )
    engine = BacktestEngine(
        allocator=allocator,
        order_sizer=OrderSizer(),
        exit_rules=[],
        constraints=constraints,
        train_pct=1.0,
        enable_split=False,
    )
    portfolio = PortfolioConfig(
        holdings=[
            Holding(ticker="A", shares=10, cost_basis=100.0),
            Holding(ticker="B", shares=10, cost_basis=100.0),
        ],
        available_cash=1_000.0,
    )
    # Sustained ~3% daily drops over 20 sessions starting day 30 → portfolio
    # bottoms around day 50 with a >40% drawdown, then stays flat.
    drops: list[float] = [0.0] * 30 + [-0.03] * 20 + [0.0] * 150
    price_data: dict[str, pd.DataFrame] = {
        "A": make_price_frame(start=date(2024, 1, 1), days=200, base_price=100.0, daily_returns=drops, name="A"),
        "B": make_price_frame(start=date(2024, 1, 1), days=200, base_price=100.0, daily_returns=drops, name="B"),
    }
    return engine.run(portfolio, price_data, start=date(2024, 1, 1), end=date(2024, 8, 30))


def test_cppi_telemetry_populates_through_engine() -> None:
    """CPPI scale-down must reach RiskMetrics, not be silently dropped.

    The backtest construct produces a deep drawdown that should drive the
    CPPI overlay to its floor. If ``risk_telemetry`` is dropped on either
    of the engine's allocation-result reconstructions (clamped or
    rebalanced), this assertion fails because every CPPI field reverts to
    its inert default (1.0 for scale, 0.0 for active_pct).
    """
    risk = RiskConfig(
        weighting="equal",
        vol_lookback_days=30,
        drawdown_penalty=2.0,
        drawdown_floor=0.4,
    )
    result = _run_backtest(risk)

    metrics = result.risk_metrics  # type: ignore[attr-defined]
    assert metrics is not None, "RiskMetrics should populate when risk: configured"

    # CPPI fired on enough bars to register: drawdown is sustained for ~150
    # bars after the drop. cppi_active_pct counts bars where exposure_scale
    # < 1.0 (i.e., drawdown > 0).
    assert metrics.cppi_active_pct > 0.3, (
        f"CPPI should be active on most bars after drawdown; got {metrics.cppi_active_pct:.2%}"
    )
    # Min scale should hit the floor (0.4) since drawdown exceeds 1/penalty = 0.5.
    assert metrics.cppi_min_scale <= 0.5, (
        f"CPPI should bind at floor under deep drawdown; got cppi_min_scale={metrics.cppi_min_scale:.3f}"
    )
    # Average gross exposure must be positive — pre-fix this was 0.0 because
    # telemetry was dropped before the per-bar history populated.
    assert metrics.avg_gross_exposure > 0, (
        f"avg_gross_exposure should be positive in any non-empty backtest; got {metrics.avg_gross_exposure}"
    )


def test_vol_target_telemetry_populates_through_engine() -> None:
    """Vol-target activity must reach RiskMetrics regardless of whether it binds.

    Configured-but-non-binding still produces ``vol_target=0.X`` in metrics
    and `vol_target_avg_scale=1.0`. We assert the *configuration* survives
    the engine round-trip — the binding test is in test_allocator_risk.
    """
    risk = RiskConfig(
        weighting="equal",
        vol_lookback_days=30,
        vol_target=0.20,
    )
    result = _run_backtest(risk)
    metrics = result.risk_metrics  # type: ignore[attr-defined]
    assert metrics is not None
    assert metrics.vol_target == 0.20, f"vol_target config should round-trip through engine; got {metrics.vol_target}"


def test_telemetry_populated_when_risk_disabled() -> None:
    """Always-on telemetry: no risk: block still produces RiskMetrics.

    Spec line 62: the bit-for-bit guarantee covers allocation outputs only;
    RiskMetrics is a new always-present output regardless of risk:
    configuration. Drawdown and Sharpe should be computed from actual
    portfolio history. (Realized-vol is a trailing-60-bar window and may
    legitimately be 0 if the last 60 bars are flat — we don't rely on it.)
    """
    result = _run_backtest(RiskConfig())
    metrics = result.risk_metrics  # type: ignore[attr-defined]
    assert metrics is not None, "RiskMetrics should populate even when risk: is omitted"
    # vol_target is None when not configured; cppi_active_pct == 0 because
    # drawdown_penalty isn't set.
    assert metrics.vol_target is None
    assert metrics.cppi_active_pct == 0.0
    # Drawdown and Sharpe should reflect actual portfolio history regardless
    # of risk configuration — these are the always-on always-meaningful fields.
    assert metrics.drawdown_from_peak > 0, (
        f"drawdown_from_peak should reflect realized portfolio DD; got {metrics.drawdown_from_peak}"
    )
    assert metrics.avg_gross_exposure > 0, (
        f"avg_gross_exposure should be positive in any non-empty backtest; "
        f"got {metrics.avg_gross_exposure} (would be 0 if telemetry plumbing is broken)"
    )
