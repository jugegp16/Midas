"""Tests for individual strategies — score() interface."""

import pandas as pd

from midas.models import StrategyTier
from midas.strategies.bollinger_band import BollingerBand
from midas.strategies.dca import DollarCostAveraging
from midas.strategies.gap_down_recovery import GapDownRecovery
from midas.strategies.ma_crossover import MovingAverageCrossover
from midas.strategies.macd_crossover import MACDCrossover
from midas.strategies.mean_reversion import MeanReversion
from midas.strategies.momentum import Momentum
from midas.strategies.profit_taking import ProfitTaking
from midas.strategies.rsi_overbought import RSIOverbought
from midas.strategies.rsi_oversold import RSIOversold
from midas.strategies.stop_loss import StopLoss
from midas.strategies.trailing_stop import TrailingStop
from midas.strategies.vwap_reversion import VWAPReversion


class TestMeanReversion:
    def test_no_signal_on_flat_prices(self, flat_prices: pd.Series) -> None:
        strategy = MeanReversion(window=30, threshold=0.10)
        assert strategy.score("FLAT", flat_prices) == 0.0

    def test_positive_score_on_drop(self, dropping_prices: pd.Series) -> None:
        strategy = MeanReversion(window=30, threshold=0.05)
        score = strategy.score("DROP", dropping_prices)
        assert score is not None
        assert score > 0.0
        assert score <= 1.0

    def test_insufficient_history(self) -> None:
        short = pd.Series([100.0] * 5, name="SHORT")
        strategy = MeanReversion(window=30)
        assert strategy.score("SHORT", short) is None

    def test_name_and_description(self) -> None:
        s = MeanReversion(window=20, threshold=0.08)
        assert "20" in s.name
        assert "0.08" in s.name
        assert len(s.suitability) > 0
        assert s.tier == StrategyTier.CONVICTION


class TestProfitTaking:
    def test_negative_score_on_gain(self, rising_prices: pd.Series) -> None:
        strategy = ProfitTaking(gain_threshold=0.15)
        score = strategy.score("RISE", rising_prices, cost_basis=100.0)
        assert score is not None
        assert score < 0.0  # bearish (sell)

    def test_abstain_without_cost_basis(self, rising_prices: pd.Series) -> None:
        strategy = ProfitTaking(gain_threshold=0.15)
        assert strategy.score("RISE", rising_prices) is None
        assert strategy.score("RISE", rising_prices, cost_basis=None) is None

    def test_neutral_below_threshold(self, flat_prices: pd.Series) -> None:
        strategy = ProfitTaking(gain_threshold=0.20)
        score = strategy.score("FLAT", flat_prices, cost_basis=100.0)
        assert score == 0.0

    def test_tier(self) -> None:
        assert ProfitTaking().tier == StrategyTier.CONVICTION


class TestMomentum:
    def test_positive_score_on_crossover(self, crossover_prices: pd.Series) -> None:
        strategy = Momentum(window=20)
        found_signal = False
        for i in range(21, len(crossover_prices)):
            score = strategy.score("CROSS", crossover_prices.iloc[: i + 1])
            if score is not None and score > 0:
                found_signal = True
                break
        assert found_signal, "Expected a positive momentum score"

    def test_neutral_on_flat(self, flat_prices: pd.Series) -> None:
        strategy = Momentum(window=20)
        assert strategy.score("FLAT", flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = pd.Series([100.0] * 10, name="SHORT")
        strategy = Momentum(window=20)
        assert strategy.score("SHORT", short) is None


class TestRSIOversold:
    def test_positive_score_on_oversold(
        self, volatile_dropping_prices: pd.Series
    ) -> None:
        strategy = RSIOversold(window=14, oversold_threshold=40.0)
        score = strategy.score("VDROP", volatile_dropping_prices)
        assert score is not None
        assert 0.0 < score <= 1.0

    def test_neutral_on_flat(self, flat_prices: pd.Series) -> None:
        strategy = RSIOversold(window=14, oversold_threshold=30.0)
        assert strategy.score("FLAT", flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = pd.Series([100.0] * 5)
        strategy = RSIOversold(window=14)
        assert strategy.score("SHORT", short) is None

    def test_name_and_description(self) -> None:
        s = RSIOversold(window=10, oversold_threshold=25.0)
        assert "10" in s.name
        assert "25" in s.name
        assert len(s.suitability) > 0
        assert s.description


class TestRSIOverbought:
    def test_negative_score_on_overbought(
        self, volatile_rising_prices: pd.Series
    ) -> None:
        strategy = RSIOverbought(window=14, overbought_threshold=65.0)
        score = strategy.score("VRISE", volatile_rising_prices)
        assert score is not None
        assert score < 0.0

    def test_neutral_on_flat(self, flat_prices: pd.Series) -> None:
        strategy = RSIOverbought(window=14, overbought_threshold=70.0)
        assert strategy.score("FLAT", flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = pd.Series([100.0] * 5)
        strategy = RSIOverbought(window=14)
        assert strategy.score("SHORT", short) is None


class TestBollingerBand:
    def test_positive_score_on_lower_band(self, dropping_prices: pd.Series) -> None:
        strategy = BollingerBand(window=20, num_std=1.5)
        score = strategy.score("DROP", dropping_prices)
        assert score is not None
        assert score > 0.0

    def test_neutral_on_flat(self, flat_prices: pd.Series) -> None:
        strategy = BollingerBand(window=20, num_std=2.0)
        assert strategy.score("FLAT", flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = pd.Series([100.0] * 5)
        strategy = BollingerBand(window=20)
        assert strategy.score("SHORT", short) is None

    def test_name_and_description(self) -> None:
        s = BollingerBand(window=20, num_std=2.5)
        assert "20" in s.name
        assert "2.5" in s.name


class TestMACDCrossover:
    def test_positive_score_on_crossover(self, crossover_prices: pd.Series) -> None:
        strategy = MACDCrossover(fast_period=12, slow_period=26, signal_period=9)
        found_signal = False
        for i in range(35, len(crossover_prices)):
            score = strategy.score("CROSS", crossover_prices.iloc[: i + 1])
            if score is not None and score > 0:
                found_signal = True
                break
        assert found_signal, "Expected a positive MACD crossover score"

    def test_neutral_on_flat(self, flat_prices: pd.Series) -> None:
        strategy = MACDCrossover()
        assert strategy.score("FLAT", flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = pd.Series([100.0] * 10)
        strategy = MACDCrossover()
        assert strategy.score("SHORT", short) is None


class TestDollarCostAveraging:
    def test_mechanical_tier(self) -> None:
        strategy = DollarCostAveraging(frequency_days=10)
        assert strategy.tier == StrategyTier.MECHANICAL

    def test_score_returns_none(self, flat_prices: pd.Series) -> None:
        strategy = DollarCostAveraging(frequency_days=10)
        assert strategy.score("FLAT", flat_prices) is None

    def test_generates_intent_on_frequency(self, flat_prices: pd.Series) -> None:
        strategy = DollarCostAveraging(frequency_days=10)
        intents = strategy.generate_intents("FLAT", flat_prices.iloc[:10])
        assert len(intents) == 1
        assert intents[0].target_value == 500.0

    def test_no_intent_off_frequency(self, flat_prices: pd.Series) -> None:
        strategy = DollarCostAveraging(frequency_days=10)
        intents = strategy.generate_intents("FLAT", flat_prices.iloc[:11])
        assert intents == []

    def test_name_and_description(self) -> None:
        s = DollarCostAveraging(frequency_days=7)
        assert "7" in s.name
        assert s.description


class TestGapDownRecovery:
    def test_positive_score_on_gap_recovery(
        self, gap_down_recovery_prices: pd.Series
    ) -> None:
        strategy = GapDownRecovery(gap_threshold=0.03)
        found_signal = False
        for i in range(3, len(gap_down_recovery_prices)):
            score = strategy.score(
                "GAP", gap_down_recovery_prices.iloc[: i + 1]
            )
            if score is not None and score > 0:
                found_signal = True
                break
        assert found_signal, "Expected a positive gap-down recovery score"

    def test_neutral_on_flat(self, flat_prices: pd.Series) -> None:
        strategy = GapDownRecovery(gap_threshold=0.03)
        assert strategy.score("FLAT", flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = pd.Series([100.0, 95.0])
        strategy = GapDownRecovery()
        assert strategy.score("SHORT", short) is None


class TestTrailingStop:
    def test_negative_score_on_drawdown(self, peak_then_drop_prices: pd.Series) -> None:
        strategy = TrailingStop(trail_pct=0.08)
        score = strategy.score(
            "PEAK", peak_then_drop_prices, cost_basis=100.0
        )
        assert score is not None
        assert score < 0.0

    def test_abstain_without_cost_basis(
        self, peak_then_drop_prices: pd.Series
    ) -> None:
        strategy = TrailingStop(trail_pct=0.08)
        assert strategy.score("PEAK", peak_then_drop_prices) is None

    def test_neutral_on_rising(self, rising_prices: pd.Series) -> None:
        strategy = TrailingStop(trail_pct=0.10)
        assert strategy.score("RISE", rising_prices, cost_basis=100.0) == 0.0

    def test_tier(self) -> None:
        assert TrailingStop().tier == StrategyTier.PROTECTIVE

    def test_name_and_description(self) -> None:
        s = TrailingStop(trail_pct=0.15)
        assert "0.15" in s.name
        assert s.description


class TestStopLoss:
    def test_negative_score_on_loss(self, dropping_prices: pd.Series) -> None:
        strategy = StopLoss(loss_threshold=0.10)
        score = strategy.score("DROP", dropping_prices, cost_basis=100.0)
        assert score is not None
        assert score < 0.0

    def test_abstain_without_cost_basis(
        self, dropping_prices: pd.Series
    ) -> None:
        strategy = StopLoss(loss_threshold=0.10)
        assert strategy.score("DROP", dropping_prices) is None

    def test_neutral_above_cost(self, rising_prices: pd.Series) -> None:
        strategy = StopLoss(loss_threshold=0.10)
        assert strategy.score("RISE", rising_prices, cost_basis=100.0) == 0.0

    def test_tier(self) -> None:
        assert StopLoss().tier == StrategyTier.PROTECTIVE

    def test_name_and_description(self) -> None:
        s = StopLoss(loss_threshold=0.05)
        assert "0.05" in s.name
        assert s.description


class TestVWAPReversion:
    def test_positive_score_below_average(self, dropping_prices: pd.Series) -> None:
        strategy = VWAPReversion(window=20, threshold=0.01)
        score = strategy.score("DROP", dropping_prices)
        assert score is not None
        assert score > 0.0

    def test_negative_score_above_average(self, rising_prices: pd.Series) -> None:
        strategy = VWAPReversion(window=20, threshold=0.01)
        score = strategy.score("RISE", rising_prices)
        assert score is not None
        assert score < 0.0

    def test_neutral_on_flat(self, flat_prices: pd.Series) -> None:
        strategy = VWAPReversion(window=20, threshold=0.02)
        assert strategy.score("FLAT", flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = pd.Series([100.0] * 5)
        strategy = VWAPReversion(window=20)
        assert strategy.score("SHORT", short) is None


class TestMovingAverageCrossover:
    def test_positive_score_on_golden_cross(
        self, ma_crossover_prices: pd.Series
    ) -> None:
        strategy = MovingAverageCrossover(short_window=10, long_window=30)
        found_signal = False
        for i in range(31, len(ma_crossover_prices)):
            score = strategy.score(
                "MACROSS", ma_crossover_prices.iloc[: i + 1]
            )
            if score is not None and score > 0:
                found_signal = True
                break
        assert found_signal, "Expected a positive golden cross score"

    def test_neutral_on_flat(self, flat_prices: pd.Series) -> None:
        strategy = MovingAverageCrossover(short_window=10, long_window=30)
        assert strategy.score("FLAT", flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = pd.Series([100.0] * 20)
        strategy = MovingAverageCrossover(short_window=10, long_window=50)
        assert strategy.score("SHORT", short) is None

    def test_name_and_description(self) -> None:
        s = MovingAverageCrossover(short_window=15, long_window=45)
        assert "15" in s.name
        assert "45" in s.name
