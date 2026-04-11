"""Tests for individual strategies — entry score() and exit clamp_target() interfaces."""

import numpy as np

from midas.strategies.base import MIN_WARMUP_CALENDAR_DAYS, warmup_bars_to_calendar_days
from midas.strategies.bollinger_band import BollingerBand
from midas.strategies.chandelier_stop import ChandelierStop
from midas.strategies.gap_down_recovery import GapDownRecovery
from midas.strategies.ma_crossover import MovingAverageCrossover
from midas.strategies.ma_crossover_exit import MovingAverageCrossoverExit
from midas.strategies.macd_crossover import MACDCrossover
from midas.strategies.macd_exit import MACDExit
from midas.strategies.mean_reversion import MeanReversion
from midas.strategies.momentum import Momentum
from midas.strategies.profit_taking import ProfitTaking
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
        assert 0.0 < score <= 1.0

    def test_insufficient_history(self) -> None:
        short = np.array([100.0] * 5)
        strategy = MeanReversion(window=30)
        assert strategy.score(short) is None

    def test_name_and_description(self) -> None:
        s = MeanReversion(window=20, threshold=0.08)
        assert s.name == "MeanReversion"
        assert len(s.suitability) > 0


class TestProfitTaking:
    def test_clamps_to_zero_on_gain(self, rising_prices: np.ndarray) -> None:
        rule = ProfitTaking(gain_threshold=0.15)
        # rising_prices ramps from 100 to ~150 — well above 15% gain.
        assert rule.clamp_target("X", 0.10, rising_prices, cost_basis=100.0, high_water_mark=150.0) == 0.0

    def test_no_clamp_below_threshold(self, flat_prices: np.ndarray) -> None:
        rule = ProfitTaking(gain_threshold=0.20)
        assert rule.clamp_target("X", 0.10, flat_prices, cost_basis=100.0, high_water_mark=100.0) == 0.10

    def test_no_clamp_without_cost_basis(self, rising_prices: np.ndarray) -> None:
        rule = ProfitTaking(gain_threshold=0.15)
        assert rule.clamp_target("X", 0.10, rising_prices, cost_basis=0.0, high_water_mark=150.0) == 0.10


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
        assert s.description


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


class TestMACDExit:
    def test_no_clamp_on_flat(self, flat_prices: np.ndarray) -> None:
        rule = MACDExit()
        assert rule.clamp_target("X", 0.10, flat_prices, cost_basis=100.0, high_water_mark=100.0) == 0.10


class TestMovingAverageCrossoverExit:
    def test_no_clamp_on_flat(self, flat_prices: np.ndarray) -> None:
        rule = MovingAverageCrossoverExit(short_window=10, long_window=30)
        assert rule.clamp_target("X", 0.10, flat_prices, cost_basis=100.0, high_water_mark=100.0) == 0.10


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
    def test_clamps_on_drawdown(self, peak_then_drop_prices: np.ndarray) -> None:
        peak = float(peak_then_drop_prices.max())
        rule = TrailingStop(trail_pct=0.08)
        # Current price is below peak and above basis — gain-protection fires.
        result = rule.clamp_target("X", 0.10, peak_then_drop_prices, cost_basis=100.0, high_water_mark=peak)
        assert result == 0.0

    def test_no_clamp_on_rising(self, rising_prices: np.ndarray) -> None:
        rule = TrailingStop(trail_pct=0.10)
        peak = float(rising_prices.max())
        # Current price == peak, so drawdown is 0% — no clamp.
        result = rule.clamp_target("X", 0.10, rising_prices, cost_basis=100.0, high_water_mark=peak)
        assert result == 0.10

    def test_name_and_description(self) -> None:
        s = TrailingStop(trail_pct=0.15)
        assert s.name == "TrailingStop"
        assert s.description

    def test_no_clamp_when_underwater(self) -> None:
        """TrailingStop is gain-protection only — no fire when losing."""
        prices = np.array([80.0, 100.0, 70.0])
        rule = TrailingStop(trail_pct=0.10)
        # Basis 80 > current 70: underwater, so no fire despite drawdown.
        result = rule.clamp_target("X", 0.10, prices, cost_basis=80.0, high_water_mark=100.0)
        assert result == 0.10


class TestChandelierStop:
    """Volatility-adjusted trailing stop: k x ATR below the rolling N-bar high."""

    def test_fires_when_drawdown_exceeds_k_atr(self) -> None:
        # 21 bars rising by $1 (101..121), then a 5-point jump to 125 (peak),
        # then a 15-point crash to 110. Last 22 bars cover everything after bar 0.
        # |diffs| over the window: 20 ones + 4 (one of the ones absorbed) + 15 = ...
        # Concretely: close-to-close ATR ≈ 1.86, highest = 125, stop ≈ 119.4.
        # Current = 110 < 119.4 → fire.
        prices = np.concatenate(
            [
                np.arange(100.0, 121.0, 1.0),  # 21 bars: 100..120
                np.array([125.0, 110.0]),  # peak then crash
            ]
        )
        rule = ChandelierStop(window=22, multiplier=3.0)
        result = rule.clamp_target("X", 0.10, prices, cost_basis=100.0, high_water_mark=125.0)
        assert result == 0.0

    def test_no_fire_on_steady_rise(self) -> None:
        # All-increasing prices: current == highest, no drawdown at all.
        prices = np.linspace(100.0, 125.0, 25)
        rule = ChandelierStop(window=22, multiplier=3.0)
        result = rule.clamp_target("X", 0.10, prices, cost_basis=100.0, high_water_mark=125.0)
        assert result == 0.10

    def test_fires_even_when_underwater(self) -> None:
        # Chandelier does not gate on cost basis the way TrailingStop does —
        # replacing StopLoss is part of the point. Peak barely clears basis,
        # then price crashes through basis: Chandelier should still fire.
        prices = np.concatenate(
            [
                np.arange(100.0, 121.0, 1.0),  # 21 bars: 100..120
                np.array([122.0, 95.0]),
            ]
        )
        rule = ChandelierStop(window=22, multiplier=3.0)
        # Basis 120 is above current 95, so position is underwater.
        result = rule.clamp_target("X", 0.10, prices, cost_basis=120.0, high_water_mark=122.0)
        assert result == 0.0

    def test_higher_multiplier_widens_stop(self) -> None:
        # Same price path; raising k should push the stop past the current
        # price and suppress the fire.
        prices = np.concatenate(
            [
                np.arange(100.0, 121.0, 1.0),
                np.array([125.0, 110.0]),
            ]
        )
        tight = ChandelierStop(window=22, multiplier=3.0)
        wide = ChandelierStop(window=22, multiplier=15.0)
        assert tight.clamp_target("X", 0.10, prices, cost_basis=100.0, high_water_mark=125.0) == 0.0
        assert wide.clamp_target("X", 0.10, prices, cost_basis=100.0, high_water_mark=125.0) == 0.10

    def test_returns_proposed_on_insufficient_history(self) -> None:
        short = np.array([100.0, 101.0, 102.0])
        rule = ChandelierStop(window=22, multiplier=3.0)
        result = rule.clamp_target("X", 0.10, short, cost_basis=100.0, high_water_mark=102.0)
        assert result == 0.10

    def test_name_and_description(self) -> None:
        rule = ChandelierStop(window=22, multiplier=3.0)
        assert rule.name == "ChandelierStop"
        assert rule.description


class TestStopLoss:
    def test_clamps_on_loss(self, dropping_prices: np.ndarray) -> None:
        rule = StopLoss(loss_threshold=0.10)
        result = rule.clamp_target("X", 0.10, dropping_prices, cost_basis=100.0, high_water_mark=100.0)
        assert result == 0.0

    def test_no_clamp_above_cost(self, rising_prices: np.ndarray) -> None:
        rule = StopLoss(loss_threshold=0.10)
        result = rule.clamp_target("X", 0.10, rising_prices, cost_basis=100.0, high_water_mark=150.0)
        assert result == 0.10

    def test_no_clamp_without_cost_basis(self, dropping_prices: np.ndarray) -> None:
        rule = StopLoss(loss_threshold=0.10)
        result = rule.clamp_target("X", 0.10, dropping_prices, cost_basis=0.0, high_water_mark=100.0)
        assert result == 0.10

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

    def test_zero_score_above_average(self, rising_prices: np.ndarray) -> None:
        # Buy-only signals clamp at 0 above the band (no negative scores).
        strategy = VWAPReversion(window=20, threshold=0.01)
        score = strategy.score(rising_prices)
        assert score is not None
        assert score == 0.0

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


class TestWarmupPeriod:
    """Each strategy advertises the history it needs before scoring."""

    def test_rolling_window_strategies_match_their_window(self) -> None:
        assert Momentum(window=25).warmup_period == 25
        assert MeanReversion(window=40).warmup_period == 40
        assert BollingerBand(window=30).warmup_period == 30
        assert VWAPReversion(window=15).warmup_period == 15

    def test_ma_crossover_uses_long_window(self) -> None:
        assert MovingAverageCrossover(short_window=10, long_window=60).warmup_period == 60

    def test_rsi_applies_recursive_multiplier(self) -> None:
        # Recursive indicators need 4x their nominal period to converge
        # (TA-Lib unstable-period convention).
        assert RSIOversold(window=14).warmup_period == 56

    def test_macd_uses_slow_period_times_multiplier_plus_signal(self) -> None:
        assert MACDCrossover(slow_period=26, signal_period=9).warmup_period == 26 * 4 + 9

    def test_stateless_exit_rules_need_no_warmup(self) -> None:
        # StopLoss / ProfitTaking / TrailingStop are price-vs-basis only.
        assert StopLoss().warmup_period == 0
        assert ProfitTaking().warmup_period == 0
        assert TrailingStop().warmup_period == 0

    def test_chandelier_stop_uses_window(self) -> None:
        assert ChandelierStop(window=22).warmup_period == 22
        assert ChandelierStop(window=30).warmup_period == 30

    def test_gap_down_recovery_needs_three_bars(self) -> None:
        assert GapDownRecovery().warmup_period == 3


class TestWarmupBarsToCalendarDays:
    """``warmup_bars_to_calendar_days`` floors at ``MIN_WARMUP_CALENDAR_DAYS``."""

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
