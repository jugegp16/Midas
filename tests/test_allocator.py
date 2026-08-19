"""Tests for the Allocator."""

from __future__ import annotations

import numpy as np
import pytest
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


class TestAllocatorRiskConfigPlumbing:
    """Threading-only tests. RiskConfig is wired through but the allocator
    behaviour with the default config must match the no-config baseline.
    """

    def test_default_risk_config_matches_no_config(self) -> None:
        mr = MeanReversion(window=5, threshold=0.01)
        constraints = AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.50)
        prices = {"A": _prices([100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 85.0]), "B": _prices([100.0] * 7)}

        baseline = Allocator([(mr, 1.0)], constraints, n_tickers=2)
        with_default_risk = Allocator([(mr, 1.0)], constraints, n_tickers=2, risk_config=RiskConfig())

        r1 = baseline.allocate(["A", "B"], prices)
        r2 = with_default_risk.allocate(["A", "B"], prices)
        assert r1.targets == r2.targets
        assert r1.blended_scores == r2.blended_scores

    def test_risk_config_property_round_trips(self) -> None:
        mr = MeanReversion(window=5, threshold=0.01)
        constraints = AllocationConstraints()
        risk = RiskConfig(weighting="inverse_vol", vol_target=0.20)
        allocator = Allocator([(mr, 1.0)], constraints, n_tickers=2, risk_config=risk)
        assert allocator.risk_config is risk

    def test_current_drawdown_kwarg_accepted(self) -> None:
        # Phase 4a implementation lands later; for now this just guards the API.
        mr = MeanReversion(window=5, threshold=0.01)
        allocator = Allocator([(mr, 1.0)], AllocationConstraints(), n_tickers=2)
        prices = {"A": _prices([100.0] * 10), "B": _prices([100.0] * 10)}
        result = allocator.allocate(["A", "B"], prices, current_drawdown=0.20)
        # Without a configured CPPI overlay, current_drawdown is a no-op.
        assert sum(result.targets.values()) > 0


class FixedScore:
    """EntrySignal stub returning a fixed per-ticker score (keyed by first close)."""

    def __init__(self, name: str, scores_by_price: dict[float, float | None]) -> None:
        self._name = name
        self._scores = scores_by_price

    @property
    def name(self) -> str:
        return self._name

    @property
    def warmup_period(self) -> int:
        return 1

    def precompute(self, price_history: PriceHistory) -> np.ndarray | None:
        return None

    def score(self, price_history: PriceHistory, **kwargs: object) -> float | None:
        return self._scores[float(price_history.close[0])]


def _fixed_prices(keys: list[float]) -> dict[str, PriceHistory]:
    return {f"T{i}": _prices([key] * 5) for i, key in enumerate(keys)}


class TestQuantileRank:
    def test_ranks_positives_and_keeps_zeros(self):
        from midas.allocator import quantile_rank

        normalized = quantile_rank({"A": 0.9, "B": 0.05, "C": 0.0, "D": 0.4})

        assert normalized["A"] == pytest.approx(3 / 3)
        assert normalized["D"] == pytest.approx(2 / 3)
        assert normalized["B"] == pytest.approx(1 / 3)
        assert normalized["C"] == 0.0  # no opinion stays no opinion

    def test_ties_share_average_rank(self):
        from midas.allocator import quantile_rank

        normalized = quantile_rank({"A": 0.5, "B": 0.5, "C": 0.9})

        assert normalized["A"] == normalized["B"] == pytest.approx(1.5 / 3)
        assert normalized["C"] == pytest.approx(1.0)

    def test_single_positive_is_top_rank(self):
        from midas.allocator import quantile_rank

        assert quantile_rank({"A": 0.03, "B": 0.0}) == {"A": 1.0, "B": 0.0}

    def test_empty_and_all_zero(self):
        from midas.allocator import quantile_rank

        assert quantile_rank({}) == {}
        assert quantile_rank({"A": 0.0}) == {"A": 0.0}


class TestForecastScaling:
    """Issue #56 acceptance: rules with different score scales get equal say."""

    def _allocator(self, forecast_scaling: str) -> Allocator:
        # Rule "timid" scores in [0, 0.1]; rule "bold" scores in [0, 1].
        # They disagree: timid's top pick is T0, bold's is T1.
        timid = FixedScore("Timid", {1.0: 0.10, 2.0: 0.02})
        bold = FixedScore("Bold", {1.0: 0.20, 2.0: 1.00})
        constraints = AllocationConstraints(min_cash_pct=0.0, forecast_scaling=forecast_scaling)
        return Allocator([(timid, 1.0), (bold, 1.0)], constraints, n_tickers=2)  # type: ignore[list-item]

    def test_raw_blend_lets_bold_rule_dominate(self):
        result = self._allocator("none").allocate(["T0", "T1"], _fixed_prices([1.0, 2.0]))

        # Raw average: T0 = (0.10+0.20)/2 = 0.15, T1 = (0.02+1.00)/2 = 0.51.
        assert result.blended_scores["T1"] > result.blended_scores["T0"]

    def test_quantile_blend_gives_rules_equal_say(self):
        result = self._allocator("quantile").allocate(["T0", "T1"], _fixed_prices([1.0, 2.0]))

        # Each rule ranks its two picks {0.5, 1.0} in opposite order, so the
        # disagreeing rules cancel exactly: both tickers blend to 0.75.
        assert result.blended_scores["T0"] == pytest.approx(0.75)
        assert result.blended_scores["T1"] == pytest.approx(0.75)
        assert result.contributions["T0"] == {"Timid": 1.0, "Bold": 0.5}
        assert result.contributions["T1"] == {"Timid": 0.5, "Bold": 1.0}

    def test_entry_weights_still_apply_after_scaling(self):
        timid = FixedScore("Timid", {1.0: 0.10, 2.0: 0.02})
        bold = FixedScore("Bold", {1.0: 0.20, 2.0: 1.00})
        constraints = AllocationConstraints(min_cash_pct=0.0, forecast_scaling="quantile")
        allocator = Allocator([(timid, 3.0), (bold, 1.0)], constraints, n_tickers=2)  # type: ignore[list-item]

        result = allocator.allocate(["T0", "T1"], _fixed_prices([1.0, 2.0]))

        # Weight 3 on Timid tilts the blend toward Timid's ranking.
        assert result.blended_scores["T0"] == pytest.approx((3 * 1.0 + 1 * 0.5) / 4)
        assert result.blended_scores["T1"] == pytest.approx((3 * 0.5 + 1 * 1.0) / 4)

    def test_abstaining_ticker_absent_from_rule_ranking(self):
        rule = FixedScore("Rule", {1.0: 0.4, 2.0: None, 3.0: 0.8})
        constraints = AllocationConstraints(min_cash_pct=0.0, forecast_scaling="quantile")
        allocator = Allocator([(rule, 1.0)], constraints, n_tickers=3)  # type: ignore[list-item]

        result = allocator.allocate(["T0", "T1", "T2"], _fixed_prices([1.0, 2.0, 3.0]))

        # T1 abstained: not ranked, contributes nothing, held.
        assert result.contributions["T1"] == {}
        assert result.contributions["T0"]["Rule"] == pytest.approx(0.5)
        assert result.contributions["T2"]["Rule"] == pytest.approx(1.0)
