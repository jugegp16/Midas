"""Tests for individual strategies — score() interface."""

import numpy as np

from midas.models import StrategyTier
from midas.strategies.base import MIN_WARMUP_CALENDAR_DAYS, warmup_bars_to_calendar_days
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
    def test_no_signal_on_flat_prices(self, flat_prices: np.ndarray) -> None:
        strategy = MeanReversion(window=30, threshold=0.10)
        assert strategy.score(flat_prices) == 0.0

    def test_positive_score_on_drop(self, dropping_prices: np.ndarray) -> None:
        strategy = MeanReversion(window=30, threshold=0.05)
        score = strategy.score(dropping_prices)
        assert score is not None
        assert score > 0.0
        assert score <= 1.0

    def test_insufficient_history(self) -> None:
        short = np.array([100.0] * 5)
        strategy = MeanReversion(window=30)
        assert strategy.score(short) is None

    def test_name_and_description(self) -> None:
        s = MeanReversion(window=20, threshold=0.08)
        assert s.name == "MeanReversion"
        assert len(s.suitability) > 0
        assert s.tier == StrategyTier.CONVICTION


class TestProfitTaking:
    def test_negative_score_on_gain(self, rising_prices: np.ndarray) -> None:
        strategy = ProfitTaking(gain_threshold=0.15)
        score = strategy.score(rising_prices, cost_basis=100.0)
        assert score is not None
        assert score < 0.0  # bearish (sell)

    def test_abstain_without_cost_basis(self, rising_prices: np.ndarray) -> None:
        strategy = ProfitTaking(gain_threshold=0.15)
        assert strategy.score(rising_prices) is None
        assert strategy.score(rising_prices, cost_basis=None) is None

    def test_neutral_below_threshold(self, flat_prices: np.ndarray) -> None:
        strategy = ProfitTaking(gain_threshold=0.20)
        score = strategy.score(flat_prices, cost_basis=100.0)
        assert score == 0.0

    def test_tier(self) -> None:
        assert ProfitTaking().tier == StrategyTier.CONVICTION


class TestMomentum:
    def test_positive_score_on_crossover(self, crossover_prices: np.ndarray) -> None:
        strategy = Momentum(window=20)
        found_signal = False
        for i in range(21, len(crossover_prices)):
            score = strategy.score(crossover_prices[: i + 1])
            if score is not None and score > 0:
                found_signal = True
                break
        assert found_signal, "Expected a positive momentum score"

    def test_neutral_on_flat(self, flat_prices: np.ndarray) -> None:
        strategy = Momentum(window=20)
        assert strategy.score(flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = np.array([100.0] * 10)
        strategy = Momentum(window=20)
        assert strategy.score(short) is None


class TestRSIOversold:
    def test_positive_score_on_oversold(self, volatile_dropping_prices: np.ndarray) -> None:
        strategy = RSIOversold(window=14, oversold_threshold=40.0)
        score = strategy.score(volatile_dropping_prices)
        assert score is not None
        assert 0.0 < score <= 1.0

    def test_neutral_on_flat(self, flat_prices: np.ndarray) -> None:
        strategy = RSIOversold(window=14, oversold_threshold=30.0)
        assert strategy.score(flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = np.array([100.0] * 5)
        strategy = RSIOversold(window=14)
        assert strategy.score(short) is None

    def test_name_and_description(self) -> None:
        s = RSIOversold(window=10, oversold_threshold=25.0)
        assert s.name == "RSIOversold"
        assert len(s.suitability) > 0
        assert s.description


class TestRSIOverbought:
    def test_negative_score_on_overbought(self, volatile_rising_prices: np.ndarray) -> None:
        strategy = RSIOverbought(window=14, overbought_threshold=65.0)
        score = strategy.score(volatile_rising_prices)
        assert score is not None
        assert score < 0.0

    def test_neutral_on_flat(self, flat_prices: np.ndarray) -> None:
        strategy = RSIOverbought(window=14, overbought_threshold=70.0)
        assert strategy.score(flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = np.array([100.0] * 5)
        strategy = RSIOverbought(window=14)
        assert strategy.score(short) is None


class TestBollingerBand:
    def test_positive_score_on_lower_band(self, dropping_prices: np.ndarray) -> None:
        strategy = BollingerBand(window=20, num_std=1.5)
        score = strategy.score(dropping_prices)
        assert score is not None
        assert score > 0.0

    def test_neutral_on_flat(self, flat_prices: np.ndarray) -> None:
        strategy = BollingerBand(window=20, num_std=2.0)
        assert strategy.score(flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = np.array([100.0] * 5)
        strategy = BollingerBand(window=20)
        assert strategy.score(short) is None

    def test_name_and_description(self) -> None:
        s = BollingerBand(window=20, num_std=2.5)
        assert s.name == "BollingerBand"


class TestMACDCrossover:
    def test_positive_score_on_crossover(self, crossover_prices: np.ndarray) -> None:
        strategy = MACDCrossover(fast_period=12, slow_period=26, signal_period=9)
        found_signal = False
        for i in range(35, len(crossover_prices)):
            score = strategy.score(crossover_prices[: i + 1])
            if score is not None and score > 0:
                found_signal = True
                break
        assert found_signal, "Expected a positive MACD crossover score"

    def test_neutral_on_flat(self, flat_prices: np.ndarray) -> None:
        strategy = MACDCrossover()
        assert strategy.score(flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = np.array([100.0] * 10)
        strategy = MACDCrossover()
        assert strategy.score(short) is None


class TestDollarCostAveraging:
    def test_mechanical_tier(self) -> None:
        strategy = DollarCostAveraging(frequency_days=10)
        assert strategy.tier == StrategyTier.MECHANICAL

    def test_score_returns_none(self, flat_prices: np.ndarray) -> None:
        strategy = DollarCostAveraging(frequency_days=10)
        assert strategy.score(flat_prices) is None

    def test_generates_intent_on_frequency(self, flat_prices: np.ndarray) -> None:
        strategy = DollarCostAveraging(frequency_days=10)
        intents = strategy.generate_intents("FLAT", flat_prices[:10])
        assert len(intents) == 1
        assert intents[0].target_value == 500.0

    def test_no_intent_off_frequency(self, flat_prices: np.ndarray) -> None:
        strategy = DollarCostAveraging(frequency_days=10)
        intents = strategy.generate_intents("FLAT", flat_prices[:11])
        assert intents == []

    def test_name_and_description(self) -> None:
        s = DollarCostAveraging(frequency_days=7)
        assert s.name == "DollarCostAveraging"
        assert s.description


class TestGapDownRecovery:
    def test_positive_score_on_gap_recovery(self, gap_down_recovery_prices: np.ndarray) -> None:
        strategy = GapDownRecovery(gap_threshold=0.03)
        found_signal = False
        for i in range(3, len(gap_down_recovery_prices)):
            score = strategy.score(gap_down_recovery_prices[: i + 1])
            if score is not None and score > 0:
                found_signal = True
                break
        assert found_signal, "Expected a positive gap-down recovery score"

    def test_neutral_on_flat(self, flat_prices: np.ndarray) -> None:
        strategy = GapDownRecovery(gap_threshold=0.03)
        assert strategy.score(flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = np.array([100.0, 95.0])
        strategy = GapDownRecovery()
        assert strategy.score(short) is None


class TestTrailingStop:
    def test_negative_score_on_drawdown(self, peak_then_drop_prices: np.ndarray) -> None:
        strategy = TrailingStop(trail_pct=0.08)
        score = strategy.score(peak_then_drop_prices, cost_basis=100.0)
        assert score is not None
        assert score < 0.0

    def test_abstain_without_cost_basis(self, peak_then_drop_prices: np.ndarray) -> None:
        strategy = TrailingStop(trail_pct=0.08)
        assert strategy.score(peak_then_drop_prices) is None

    def test_neutral_on_rising(self, rising_prices: np.ndarray) -> None:
        strategy = TrailingStop(trail_pct=0.10)
        assert strategy.score(rising_prices, cost_basis=100.0) == 0.0

    def test_tier(self) -> None:
        assert TrailingStop().tier == StrategyTier.PROTECTIVE

    def test_name_and_description(self) -> None:
        s = TrailingStop(trail_pct=0.15)
        assert s.name == "TrailingStop"
        assert s.description


class TestStopLoss:
    def test_negative_score_on_loss(self, dropping_prices: np.ndarray) -> None:
        strategy = StopLoss(loss_threshold=0.10)
        score = strategy.score(dropping_prices, cost_basis=100.0)
        assert score is not None
        assert score < 0.0

    def test_abstain_without_cost_basis(self, dropping_prices: np.ndarray) -> None:
        strategy = StopLoss(loss_threshold=0.10)
        assert strategy.score(dropping_prices) is None

    def test_neutral_above_cost(self, rising_prices: np.ndarray) -> None:
        strategy = StopLoss(loss_threshold=0.10)
        assert strategy.score(rising_prices, cost_basis=100.0) == 0.0

    def test_tier(self) -> None:
        assert StopLoss().tier == StrategyTier.PROTECTIVE

    def test_name_and_description(self) -> None:
        s = StopLoss(loss_threshold=0.05)
        assert s.name == "StopLoss"
        assert s.description


class TestVWAPReversion:
    def test_positive_score_below_average(self, dropping_prices: np.ndarray) -> None:
        strategy = VWAPReversion(window=20, threshold=0.01)
        score = strategy.score(dropping_prices)
        assert score is not None
        assert score > 0.0

    def test_negative_score_above_average(self, rising_prices: np.ndarray) -> None:
        strategy = VWAPReversion(window=20, threshold=0.01)
        score = strategy.score(rising_prices)
        assert score is not None
        assert score < 0.0

    def test_neutral_on_flat(self, flat_prices: np.ndarray) -> None:
        strategy = VWAPReversion(window=20, threshold=0.02)
        assert strategy.score(flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = np.array([100.0] * 5)
        strategy = VWAPReversion(window=20)
        assert strategy.score(short) is None


class TestMovingAverageCrossover:
    def test_positive_score_on_golden_cross(self, ma_crossover_prices: np.ndarray) -> None:
        strategy = MovingAverageCrossover(short_window=10, long_window=30)
        found_signal = False
        for i in range(31, len(ma_crossover_prices)):
            score = strategy.score(ma_crossover_prices[: i + 1])
            if score is not None and score > 0:
                found_signal = True
                break
        assert found_signal, "Expected a positive golden cross score"

    def test_neutral_on_flat(self, flat_prices: np.ndarray) -> None:
        strategy = MovingAverageCrossover(short_window=10, long_window=30)
        assert strategy.score(flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = np.array([100.0] * 20)
        strategy = MovingAverageCrossover(short_window=10, long_window=50)
        assert strategy.score(short) is None

    def test_name_and_description(self) -> None:
        s = MovingAverageCrossover(short_window=15, long_window=45)
        assert s.name == "MovingAverageCrossover"


class TestWarmupPeriod:
    """Each strategy advertises the history it needs before scoring."""

    def test_rolling_window_strategies_match_their_window(self) -> None:
        assert Momentum(window=25).warmup_period == 25
        assert MeanReversion(window=40).warmup_period == 40
        assert BollingerBand(window=30).warmup_period == 30
        assert VWAPReversion(window=15).warmup_period == 15

    def test_ma_crossover_uses_long_window(self) -> None:
        assert MovingAverageCrossover(short_window=10, long_window=60).warmup_period == 60

    def test_rsi_warmup_is_window_plus_one(self) -> None:
        # RSI uses SMA (not Wilder EMA), so only window+1 bars are needed —
        # window bars produce window deltas, which is exactly one SMA period.
        assert RSIOversold(window=14).warmup_period == 15
        assert RSIOverbought(window=14).warmup_period == 15

    def test_macd_uses_slow_period_times_multiplier_plus_signal(self) -> None:
        assert MACDCrossover(slow_period=26, signal_period=9).warmup_period == 26 * 4 + 9

    def test_stateless_exit_strategies_need_no_warmup(self) -> None:
        # StopLoss / ProfitTaking / TrailingStop / DCA inherit the default 0.
        assert StopLoss().warmup_period == 0
        assert ProfitTaking().warmup_period == 0
        assert TrailingStop().warmup_period == 0
        assert DollarCostAveraging().warmup_period == 0

    def test_gap_down_recovery_needs_three_bars(self) -> None:
        assert GapDownRecovery().warmup_period == 3


class TestWarmupBarsToCalendarDays:
    """``warmup_bars_to_calendar_days`` floors at ``MIN_WARMUP_CALENDAR_DAYS``.

    Live mode derives its history-fetch window from this helper. If a
    mechanical-only setup (StopLoss/TrailingStop/DCA) returned 0, the
    live engine would request a single-day price history and frequently
    receive nothing on weekends/holidays. The floor prevents that.
    """

    def test_zero_bars_floors_to_minimum(self) -> None:
        assert warmup_bars_to_calendar_days(0) == MIN_WARMUP_CALENDAR_DAYS

    def test_negative_bars_floors_to_minimum(self) -> None:
        assert warmup_bars_to_calendar_days(-5) == MIN_WARMUP_CALENDAR_DAYS

    def test_small_warmup_floors_to_minimum(self) -> None:
        # 3 bars * 1.5 + 10 = 14, still below the 30-day floor.
        assert warmup_bars_to_calendar_days(3) == MIN_WARMUP_CALENDAR_DAYS

    def test_large_warmup_scales_above_floor(self) -> None:
        # 100 bars * 1.5 + 10 = 160, well above the floor.
        assert warmup_bars_to_calendar_days(100) == 160
