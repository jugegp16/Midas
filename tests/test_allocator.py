"""Tests for the Allocator."""

from __future__ import annotations

import numpy as np
from conftest import ph

from midas.allocator import Allocator
from midas.data.price_history import PriceHistory
from midas.models import AllocationConstraints, RiskConfig
from midas.strategies.mean_reversion import MeanReversion
from midas.strategies.momentum import Momentum


def _prices(values: list[float]) -> PriceHistory:
    return ph(np.array(values))


class TestAllocator:
    def test_uniform_scores_produce_equal_weights(self):
        """When all entries score 0.0, no ticker is active → held tickers fall back to base."""
        mr = MeanReversion(window=5, threshold=0.01)
        constraints = AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.50)
        allocator = Allocator([(mr, 1.0)], constraints, n_tickers=2)
        prices = {"A": _prices([100.0] * 10), "B": _prices([100.0] * 10)}
        result = allocator.allocate(["A", "B"], prices)
        # Flat prices → MeanReversion scores 0.0 → both tickers held at base.
        base = (1 - 0.05) / 2  # 0.475
        assert abs(result.targets["A"] - base) < 1e-9
        assert abs(result.targets["B"] - base) < 1e-9

    def test_bullish_score_increases_weight(self):
        """A positive blended score should produce weight > base_weight."""
        mr = MeanReversion(window=5, threshold=0.01)
        constraints = AllocationConstraints(min_cash_pct=0.05, softmax_temperature=0.5, max_position_pct=0.90)
        allocator = Allocator([(mr, 1.0)], constraints, n_tickers=2)
        # A has dropping prices (should trigger mean reversion buy -> positive score)
        prices_a = _prices([100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 85.0])
        # B has flat prices
        prices_b = _prices([100.0] * 7)
        # Pass current_weights so the held (score=0) ticker B doesn't consume
        # the base budget — exposes A's bullish score in its target weight.
        result = allocator.allocate(
            ["A", "B"],
            {"A": prices_a, "B": prices_b},
            current_weights={"A": 0.0, "B": 0.0},
        )
        base = (1 - 0.05) / 2
        assert result.targets["A"] > base
        assert result.blended_scores["A"] > 0

    def test_abstention_excluded_from_blend(self):
        """None scores are excluded from both numerator and denominator."""
        # window=100 ensures None (not enough data)
        mr = MeanReversion(window=100, threshold=0.10)
        mom = Momentum(window=5)
        constraints = AllocationConstraints(min_cash_pct=0.0)
        allocator = Allocator([(mr, 1.0), (mom, 1.0)], constraints, n_tickers=1)
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
        constraints = AllocationConstraints(max_position_pct=0.10, min_cash_pct=0.05)
        allocator = Allocator([(mr, 1.0)], constraints, n_tickers=2)
        # Extremely bullish signal
        prices_a = _prices([100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 50.0])
        prices_b = _prices([100.0] * 7)
        result = allocator.allocate(["A", "B"], {"A": prices_a, "B": prices_b})
        assert result.targets["A"] <= 0.10 + 1e-9

    def test_normalization_respects_min_cash(self):
        """Sum of all target weights should not exceed 1 - min_cash_pct."""
        mr = MeanReversion(window=5, threshold=0.01)
        constraints = AllocationConstraints(min_cash_pct=0.20, softmax_temperature=0.2)
        allocator = Allocator([(mr, 1.0)], constraints, n_tickers=3)
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
        allocator = Allocator([], constraints, 0)
        result = allocator.allocate([], {})
        assert result.targets == {}

    def test_neutral_holds_current_weight(self):
        """When no entry signal scores, target = current weight (Option A)."""
        # window too large to score
        mr = MeanReversion(window=100, threshold=0.01)
        constraints = AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.90)
        allocator = Allocator([(mr, 1.0)], constraints, n_tickers=2)
        prices = {
            "A": _prices([100.0] * 10),
            "B": _prices([100.0] * 10),
        }
        # A is overweight, B is underweight — neither should move.
        result = allocator.allocate(
            ["A", "B"],
            prices,
            current_weights={"A": 0.70, "B": 0.10},
        )
        assert abs(result.targets["A"] - 0.70) < 1e-9
        assert abs(result.targets["B"] - 0.10) < 1e-9

    def test_neutral_without_current_weights_falls_back_to_base(self):
        """Omitting current_weights falls back to equal-weight base."""
        mr = MeanReversion(window=100, threshold=0.01)
        constraints = AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.90)
        allocator = Allocator([(mr, 1.0)], constraints, n_tickers=2)
        prices = {"A": _prices([100.0] * 10), "B": _prices([100.0] * 10)}
        result = allocator.allocate(["A", "B"], prices)
        base = (1 - 0.05) / 2
        assert abs(result.targets["A"] - base) < 1e-9

    def test_softmax_sum_equals_investable(self):
        """Softmax construct-to-budget: sum of active targets == investable, exactly."""
        mr = MeanReversion(window=5, threshold=0.01)
        constraints = AllocationConstraints(min_cash_pct=0.20, softmax_temperature=0.2, max_position_pct=0.90)
        allocator = Allocator([(mr, 1.0)], constraints, n_tickers=3)
        prices = {
            "A": _prices([100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 50.0]),
            "B": _prices([100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 50.0]),
            "C": _prices([100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 50.0]),
        }
        result = allocator.allocate(["A", "B", "C"], prices)
        investable = 1.0 - 0.20
        assert abs(sum(result.targets.values()) - investable) < 1e-9

    def test_softmax_concentrates_on_higher_conviction(self):
        """Higher blended score → larger softmax share of the budget."""
        # Use a large threshold so scores don't saturate at 1.0.
        mr = MeanReversion(window=5, threshold=0.20)
        constraints = AllocationConstraints(min_cash_pct=0.05, softmax_temperature=0.25, max_position_pct=0.90)
        allocator = Allocator([(mr, 1.0)], constraints, n_tickers=3)
        # A moderately bullish (bigger drop), B mildly bullish, C flat.
        prices = {
            "A": _prices([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 90.0]),  # ~10% drop
            "B": _prices([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 96.0]),  # ~4% drop
            "C": _prices([100.0] * 7),  # flat
        }
        # A and B are active (positive scores); C is held with no current
        # weight → falls back to base. Test that A > B (active concentration).
        result = allocator.allocate(["A", "B", "C"], prices)
        assert result.targets["A"] > result.targets["B"]
        # Total respects investable budget.
        assert sum(result.targets.values()) <= 0.95 + 1e-9

    def test_cap_with_redistribution(self):
        """When a cap clamps a ticker, freed budget redistributes to survivors."""
        mr = MeanReversion(window=5, threshold=0.01)
        constraints = AllocationConstraints(min_cash_pct=0.05, softmax_temperature=0.125, max_position_pct=0.50)
        allocator = Allocator([(mr, 1.0)], constraints, n_tickers=2)
        # A overwhelmingly bullish → pre-cap softmax would give it > 0.50.
        prices_a = _prices([100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 40.0])
        prices_b = _prices([100.0] * 7)
        # Pass current_weights so B (score 0 → held) doesn't consume base
        # budget — leaves the full investable budget for A's softmax to be
        # capped against.
        result = allocator.allocate(
            ["A", "B"],
            {"A": prices_a, "B": prices_b},
            current_weights={"A": 0.0, "B": 0.0},
        )
        # A pinned at cap.
        assert abs(result.targets["A"] - 0.50) < 1e-9

    def test_cap_redistribution_multi_iteration(self):
        """Capping one ticker can push another over the cap → second loop pass.

        Regression test for the ``_apply_cap_with_redistribution`` iteration:
        with very concentrated softmax (low T) and a tight cap, pinning the
        top ticker frees enough budget to push the second-place ticker above
        the cap as well, forcing the loop to run another iteration.
        """
        # threshold=0.20 yields non-saturated MeanReversion scores.
        mr = MeanReversion(window=5, threshold=0.20)
        constraints = AllocationConstraints(
            min_cash_pct=0.05,
            softmax_temperature=0.1,  # very concentrated
            max_position_pct=0.35,
        )
        allocator = Allocator([(mr, 1.0)], constraints, n_tickers=3)
        # A: 20% drop → score saturates high
        # B: 15% drop → score mid-high (would exceed cap on pass 2)
        # C: ~5% drop → score low
        prices_a = _prices([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 80.0])
        prices_b = _prices([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 85.0])
        prices_c = _prices([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 95.0])
        result = allocator.allocate(
            ["A", "B", "C"],
            {"A": prices_a, "B": prices_b, "C": prices_c},
        )
        # Both A and B pinned at the cap — the second pin only happens if the
        # redistribution loop runs twice.
        assert abs(result.targets["A"] - 0.35) < 1e-9
        assert abs(result.targets["B"] - 0.35) < 1e-9


class TestAllocatorRiskConfig:
    def test_accepts_risk_config_kwarg(self):
        mr = MeanReversion(window=5, threshold=0.01)
        constraints = AllocationConstraints(min_cash_pct=0.05)
        risk = RiskConfig()
        allocator = Allocator([(mr, 1.0)], constraints, n_tickers=2, risk_config=risk)
        # Verify it's stored on the instance for later phases to read.
        assert allocator._risk_config is risk

    def test_defaults_to_risk_config_when_omitted(self):
        mr = MeanReversion(window=5, threshold=0.01)
        constraints = AllocationConstraints(min_cash_pct=0.05)
        allocator = Allocator([(mr, 1.0)], constraints, n_tickers=2)
        # Default matches RiskConfig() exactly.
        assert allocator._risk_config == RiskConfig()
