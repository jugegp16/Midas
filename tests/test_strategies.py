"""Tests for individual strategies — entry score() and exit evaluate_exit() interfaces."""

from datetime import date

import numpy as np

from midas.models import PositionLot
from midas.strategies.base import MIN_WARMUP_CALENDAR_DAYS, warmup_bars_to_calendar_days
from midas.strategies.bollinger_band import BollingerBand
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


def _lot(
    shares: float = 10.0,
    basis: float = 100.0,
    high_water_mark: float | None = None,
) -> PositionLot:
    return PositionLot(
        shares=shares,
        purchase_date=date(2024, 1, 1),
        cost_basis=basis,
        high_water_mark=high_water_mark,
    )


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
    def test_fires_on_gain(self, rising_prices: np.ndarray) -> None:
        strategy = ProfitTaking(gain_threshold=0.15)
        # rising_prices ramps from 100 to ~150 — well above 15% gain.
        intents = strategy.evaluate_exit("X", [_lot(basis=100.0)], rising_prices)
        assert len(intents) == 1
        assert intents[0].target_value > 0

    def test_no_fire_below_threshold(self, flat_prices: np.ndarray) -> None:
        strategy = ProfitTaking(gain_threshold=0.20)
        intents = strategy.evaluate_exit("X", [_lot(basis=100.0)], flat_prices)
        assert intents == []

    def test_no_fire_without_lots(self, rising_prices: np.ndarray) -> None:
        strategy = ProfitTaking(gain_threshold=0.15)
        assert strategy.evaluate_exit("X", [], rising_prices) == []


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
    def test_no_fire_on_flat(self, flat_prices: np.ndarray) -> None:
        rule = MACDExit()
        assert rule.evaluate_exit("X", [_lot()], flat_prices) == []


class TestMovingAverageCrossoverExit:
    def test_no_fire_on_flat(self, flat_prices: np.ndarray) -> None:
        rule = MovingAverageCrossoverExit(short_window=10, long_window=30)
        assert rule.evaluate_exit("X", [_lot()], flat_prices) == []


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
    def test_fires_on_drawdown(self, peak_then_drop_prices: np.ndarray) -> None:
        # Engine is responsible for tracking each lot's high-water mark.
        # Pre-seed it to the peak so the strategy sees the full drawdown.
        peak = float(peak_then_drop_prices.max())
        strategy = TrailingStop(trail_pct=0.08)
        intents = strategy.evaluate_exit(
            "X",
            [_lot(basis=100.0, high_water_mark=peak)],
            peak_then_drop_prices,
        )
        assert len(intents) == 1

    def test_no_fire_on_rising(self, rising_prices: np.ndarray) -> None:
        strategy = TrailingStop(trail_pct=0.10)
        peak = float(rising_prices.max())
        assert (
            strategy.evaluate_exit(
                "X",
                [_lot(basis=100.0, high_water_mark=peak)],
                rising_prices,
            )
            == []
        )

    def test_name_and_description(self) -> None:
        s = TrailingStop(trail_pct=0.15)
        assert s.name == "TrailingStop"
        assert s.description

    def test_multi_lot_only_triggered_lots_sold(self) -> None:
        """Per-lot HWM: only lots past their own drawdown threshold sell.

        Two lots in the same ticker:
          - Lot A: 10 shares, basis $80, HWM $120 — drawdown from $120 to
            current $100 is 16.7% (> 10% threshold) AND still in profit
            ($100 > $80). Triggers.
          - Lot B: 5 shares, basis $95, HWM $102 — drawdown from $102 to
            $100 is 2.0% (< 10% threshold). Does not trigger.

        Expected: one intent for 10 shares (lot A), value $1000. Lot B is
        untouched — a flat HWM across the position would have sold both.
        """
        current = 100.0
        prices = np.array([80.0, 120.0, 102.0, current])
        lots = [
            _lot(shares=10.0, basis=80.0, high_water_mark=120.0),
            _lot(shares=5.0, basis=95.0, high_water_mark=102.0),
        ]
        strategy = TrailingStop(trail_pct=0.10)
        intents = strategy.evaluate_exit("X", lots, prices)
        assert len(intents) == 1
        assert intents[0].target_value == 10.0 * current

    def test_multi_lot_losing_lot_skipped(self) -> None:
        """A lot that's underwater must not fire trailing-stop even if its
        HWM drawdown crosses the threshold — TrailingStop is gain-protection,
        StopLoss handles losses."""
        current = 70.0
        prices = np.array([80.0, 100.0, current])
        # Lot is underwater (basis 80 > current 70) but drawdown from HWM
        # 100 → 70 is 30%, well past 10% threshold.
        lots = [_lot(shares=10.0, basis=80.0, high_water_mark=100.0)]
        strategy = TrailingStop(trail_pct=0.10)
        assert strategy.evaluate_exit("X", lots, prices) == []


class TestStopLoss:
    def test_fires_on_loss(self, dropping_prices: np.ndarray) -> None:
        strategy = StopLoss(loss_threshold=0.10)
        intents = strategy.evaluate_exit("X", [_lot(basis=100.0)], dropping_prices)
        assert len(intents) == 1

    def test_no_fire_above_cost(self, rising_prices: np.ndarray) -> None:
        strategy = StopLoss(loss_threshold=0.10)
        assert strategy.evaluate_exit("X", [_lot(basis=100.0)], rising_prices) == []

    def test_no_fire_without_lots(self, dropping_prices: np.ndarray) -> None:
        strategy = StopLoss(loss_threshold=0.10)
        assert strategy.evaluate_exit("X", [], dropping_prices) == []

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
