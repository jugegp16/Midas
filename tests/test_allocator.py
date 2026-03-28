"""Tests for the Allocator."""

from __future__ import annotations

import math

import pandas as pd

from midas.allocator import Allocator
from midas.models import AllocationConstraints
from midas.strategies.mean_reversion import MeanReversion
from midas.strategies.momentum import Momentum
from midas.strategies.stop_loss import StopLoss


def _prices(values: list[float]) -> pd.Series:
    return pd.Series(values)


class TestAllocator:
    def test_uniform_scores_produce_equal_weights(self):
        """When all strategies score 0.0, targets equal base_weight (after cap)."""
        mr = MeanReversion(window=5, threshold=0.01)
        constraints = AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.50)
        allocator = Allocator(
            conviction_strategies=[(mr, 1.0)],
            protective_strategies=[],
            constraints=constraints,
            n_tickers=2,
        )
        # Flat prices -> score 0.0
        prices = {
            "A": _prices([100.0] * 10),
            "B": _prices([100.0] * 10),
        }
        result = allocator.allocate(["A", "B"], prices)
        base = (1 - 0.05) / 2  # 0.475
        # sigmoid(0) = 0.5, so target = base * 2 * 0.5 = base
        assert abs(result.targets["A"] - base) < 1e-6
        assert abs(result.targets["B"] - base) < 1e-6

    def test_bullish_score_increases_weight(self):
        """A positive blended score should produce weight > base_weight."""
        mr = MeanReversion(window=5, threshold=0.01)
        constraints = AllocationConstraints(
            min_cash_pct=0.05, sigmoid_steepness=2.0, max_position_pct=0.90
        )
        allocator = Allocator(
            conviction_strategies=[(mr, 1.0)],
            protective_strategies=[],
            constraints=constraints,
            n_tickers=2,
        )
        # A has dropping prices (should trigger mean reversion buy -> positive score)
        prices_a = _prices([100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 85.0])
        # B has flat prices
        prices_b = _prices([100.0] * 7)
        result = allocator.allocate(["A", "B"], {"A": prices_a, "B": prices_b})
        base = (1 - 0.05) / 2
        assert result.targets["A"] > base
        assert result.blended_scores["A"] > 0

    def test_protective_veto_forces_zero(self):
        """PROTECTIVE strategy can force target weight to 0."""
        mr = MeanReversion(window=5, threshold=0.50)  # won't fire
        sl = StopLoss(loss_threshold=0.05)
        constraints = AllocationConstraints(min_cash_pct=0.05)
        allocator = Allocator(
            conviction_strategies=[(mr, 1.0)],
            protective_strategies=[(sl, -0.5)],
            constraints=constraints,
            n_tickers=1,
        )
        # Price dropped 20% from cost basis -> StopLoss fires negative
        prices = {"A": _prices([100.0] * 10 + [80.0])}
        context = {"A": {"cost_basis": 100.0}}
        result = allocator.allocate(["A"], prices, context)
        assert result.targets["A"] == 0.0

    def test_abstention_excluded_from_blend(self):
        """None scores are excluded from both numerator and denominator."""
        # window=100 ensures None (not enough data)
        mr = MeanReversion(window=100, threshold=0.10)
        mom = Momentum(window=5)
        constraints = AllocationConstraints(min_cash_pct=0.0)
        allocator = Allocator(
            conviction_strategies=[(mr, 1.0), (mom, 1.0)],
            protective_strategies=[],
            constraints=constraints,
            n_tickers=1,
        )
        # Only 10 data points: MR needs 100 -> None, Momentum needs 6 -> has opinion
        prices = {"A": _prices([100.0] * 10)}
        result = allocator.allocate(["A"], prices)
        # MR should not appear in contributions (it abstained)
        assert "MeanReversion" not in str(result.contributions["A"])
        # Momentum should be the only contributor
        assert len(result.contributions["A"]) <= 1

    def test_max_position_pct_caps_weight(self):
        """Target weights are capped at max_position_pct."""
        mr = MeanReversion(window=5, threshold=0.01)
        constraints = AllocationConstraints(
            max_position_pct=0.10, min_cash_pct=0.05
        )
        allocator = Allocator(
            conviction_strategies=[(mr, 1.0)],
            protective_strategies=[],
            constraints=constraints,
            n_tickers=2,
        )
        # Extremely bullish signal
        prices_a = _prices([100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 50.0])
        prices_b = _prices([100.0] * 7)
        result = allocator.allocate(["A", "B"], {"A": prices_a, "B": prices_b})
        assert result.targets["A"] <= 0.10 + 1e-9

    def test_normalization_respects_min_cash(self):
        """Sum of all target weights should not exceed 1 - min_cash_pct."""
        mr = MeanReversion(window=5, threshold=0.01)
        constraints = AllocationConstraints(
            min_cash_pct=0.20, sigmoid_steepness=5.0
        )
        allocator = Allocator(
            conviction_strategies=[(mr, 1.0)],
            protective_strategies=[],
            constraints=constraints,
            n_tickers=3,
        )
        prices = {
            "A": _prices([100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 50.0]),
            "B": _prices([100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 50.0]),
            "C": _prices([100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 50.0]),
        }
        result = allocator.allocate(["A", "B", "C"], prices)
        total = sum(result.targets.values())
        assert total <= 1.0 - 0.20 + 1e-9

    def test_empty_tickers(self):
        """No tickers -> empty result."""
        constraints = AllocationConstraints()
        allocator = Allocator([], [], constraints, 0)
        result = allocator.allocate([], {})
        assert result.targets == {}

    def test_sigmoid_symmetry(self):
        """Verify sigmoid transform is symmetric around 0."""
        k = 2.0
        for score in [0.5, 1.0, 0.3]:
            pos = 1.0 / (1.0 + math.exp(-k * score))
            neg = 1.0 / (1.0 + math.exp(-k * (-score)))
            # pos + neg should equal 1.0 (sigmoid symmetry)
            assert abs(pos + neg - 1.0) < 1e-10
