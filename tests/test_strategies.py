"""Tests for individual strategies — entry score() and exit clamp_target() interfaces."""

import numpy as np
from conftest import ph

from midas.data.price_history import PriceHistory
from midas.strategies.base import MIN_WARMUP_CALENDAR_DAYS, warmup_bars_to_calendar_days
from midas.strategies.bollinger_band import BollingerBand
from midas.strategies.chandelier_stop import ChandelierStop
from midas.strategies.donchian_breakout import DonchianBreakout
from midas.strategies.gap_down_recovery import GapDownRecovery
from midas.strategies.keltner_channel import KeltnerChannel
from midas.strategies.ma_crossover import MovingAverageCrossover
from midas.strategies.ma_crossover_exit import MovingAverageCrossoverExit
from midas.strategies.macd_crossover import MACDCrossover
from midas.strategies.macd_exit import MACDExit
from midas.strategies.mean_reversion import MeanReversion
from midas.strategies.momentum import Momentum
from midas.strategies.parabolic_sar_exit import ParabolicSARExit
from midas.strategies.profit_taking import ProfitTaking
from midas.strategies.rsi_oversold import RSIOversold
from midas.strategies.stop_loss import StopLoss
from midas.strategies.trailing_stop import TrailingStop
from midas.strategies.vwap_reversion import VWAPReversion


class TestMeanReversion:
    def test_no_signal_on_flat_prices(self, flat_prices: PriceHistory) -> None:
        strategy = MeanReversion(window=30, threshold=0.10)
        assert strategy.score(flat_prices) == 0.0

    def test_positive_score_on_drop(self, dropping_prices: PriceHistory) -> None:
        strategy = MeanReversion(window=30, threshold=0.05)
        score = strategy.score(dropping_prices)
        assert score is not None
        assert 0.0 < score <= 1.0

    def test_insufficient_history(self) -> None:
        short = ph(np.array([100.0] * 5))
        strategy = MeanReversion(window=30)
        assert strategy.score(short) is None

    def test_name_and_description(self) -> None:
        s = MeanReversion(window=20, threshold=0.08)
        assert s.name == "MeanReversion"
        assert len(s.suitability) > 0


class TestProfitTaking:
    def test_clamps_to_zero_on_gain(self, rising_prices: PriceHistory) -> None:
        rule = ProfitTaking(gain_threshold=0.15)
        # rising_prices ramps from 100 to ~150 — well above 15% gain.
        assert rule.clamp_target("X", 0.10, rising_prices, cost_basis=100.0, high_water_mark=150.0) == 0.0

    def test_no_clamp_below_threshold(self, flat_prices: PriceHistory) -> None:
        rule = ProfitTaking(gain_threshold=0.20)
        assert rule.clamp_target("X", 0.10, flat_prices, cost_basis=100.0, high_water_mark=100.0) == 0.10

    def test_no_clamp_without_cost_basis(self, rising_prices: PriceHistory) -> None:
        rule = ProfitTaking(gain_threshold=0.15)
        assert rule.clamp_target("X", 0.10, rising_prices, cost_basis=0.0, high_water_mark=150.0) == 0.10


class TestMomentum:
    def test_positive_score_on_crossover(self, crossover_prices: PriceHistory) -> None:
        strategy = Momentum(window=20)
        found_signal = False
        for i in range(21, len(crossover_prices)):
            score = strategy.score(crossover_prices[: i + 1])
            if score is not None and score > 0:
                found_signal = True
                break
        assert found_signal, "Expected a positive momentum score"

    def test_neutral_on_flat(self, flat_prices: PriceHistory) -> None:
        strategy = Momentum(window=20)
        assert strategy.score(flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = ph(np.array([100.0] * 10))
        strategy = Momentum(window=20)
        assert strategy.score(short) is None


class TestRSIOversold:
    def test_positive_score_on_oversold(self, volatile_dropping_prices: PriceHistory) -> None:
        strategy = RSIOversold(window=14, oversold_threshold=40.0)
        score = strategy.score(volatile_dropping_prices)
        assert score is not None
        assert 0.0 < score <= 1.0

    def test_neutral_on_flat(self, flat_prices: PriceHistory) -> None:
        strategy = RSIOversold(window=14, oversold_threshold=30.0)
        assert strategy.score(flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = ph(np.array([100.0] * 5))
        strategy = RSIOversold(window=14)
        assert strategy.score(short) is None

    def test_name_and_description(self) -> None:
        s = RSIOversold(window=10, oversold_threshold=25.0)
        assert s.name == "RSIOversold"
        assert s.description


class TestBollingerBand:
    def test_positive_score_on_lower_band(self, dropping_prices: PriceHistory) -> None:
        strategy = BollingerBand(window=20, num_std=1.5)
        score = strategy.score(dropping_prices)
        assert score is not None
        assert score > 0.0

    def test_neutral_on_flat(self, flat_prices: PriceHistory) -> None:
        strategy = BollingerBand(window=20, num_std=2.0)
        assert strategy.score(flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = ph(np.array([100.0] * 5))
        strategy = BollingerBand(window=20)
        assert strategy.score(short) is None


class TestMACDCrossover:
    def test_positive_score_on_crossover(self, crossover_prices: PriceHistory) -> None:
        strategy = MACDCrossover(fast_period=12, slow_period=26, signal_period=9)
        found_signal = False
        for i in range(35, len(crossover_prices)):
            score = strategy.score(crossover_prices[: i + 1])
            if score is not None and score > 0:
                found_signal = True
                break
        assert found_signal, "Expected a positive MACD crossover score"

    def test_neutral_on_flat(self, flat_prices: PriceHistory) -> None:
        strategy = MACDCrossover()
        assert strategy.score(flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = ph(np.array([100.0] * 10))
        strategy = MACDCrossover()
        assert strategy.score(short) is None


class TestMACDExit:
    def test_no_clamp_on_flat(self, flat_prices: PriceHistory) -> None:
        rule = MACDExit()
        assert rule.clamp_target("X", 0.10, flat_prices, cost_basis=100.0, high_water_mark=100.0) == 0.10


class TestMovingAverageCrossoverExit:
    def test_no_clamp_on_flat(self, flat_prices: PriceHistory) -> None:
        rule = MovingAverageCrossoverExit(short_window=10, long_window=30)
        assert rule.clamp_target("X", 0.10, flat_prices, cost_basis=100.0, high_water_mark=100.0) == 0.10


class TestGapDownRecovery:
    def test_positive_score_on_gap_recovery(self, gap_down_recovery_prices: PriceHistory) -> None:
        strategy = GapDownRecovery(gap_threshold=0.03)
        found_signal = False
        for i in range(3, len(gap_down_recovery_prices)):
            score = strategy.score(gap_down_recovery_prices[: i + 1])
            if score is not None and score > 0:
                found_signal = True
                break
        assert found_signal, "Expected a positive gap-down recovery score"

    def test_neutral_on_flat(self, flat_prices: PriceHistory) -> None:
        strategy = GapDownRecovery(gap_threshold=0.03)
        assert strategy.score(flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = ph(np.array([100.0, 95.0]))
        strategy = GapDownRecovery()
        assert strategy.score(short) is None


class TestTrailingStop:
    def test_clamps_on_drawdown(self, peak_then_drop_prices: PriceHistory) -> None:
        peak = float(peak_then_drop_prices.close.max())
        rule = TrailingStop(trail_pct=0.08)
        # Current price is below peak and above basis — gain-protection fires.
        result = rule.clamp_target("X", 0.10, peak_then_drop_prices, cost_basis=100.0, high_water_mark=peak)
        assert result == 0.0

    def test_no_clamp_on_rising(self, rising_prices: PriceHistory) -> None:
        rule = TrailingStop(trail_pct=0.10)
        peak = float(rising_prices.close.max())
        # Current price == peak, so drawdown is 0% — no clamp.
        result = rule.clamp_target("X", 0.10, rising_prices, cost_basis=100.0, high_water_mark=peak)
        assert result == 0.10

    def test_name_and_description(self) -> None:
        s = TrailingStop(trail_pct=0.15)
        assert s.name == "TrailingStop"
        assert s.description

    def test_no_clamp_when_underwater(self) -> None:
        """TrailingStop is gain-protection only — no fire when losing."""
        prices = ph(np.array([80.0, 100.0, 70.0]))
        rule = TrailingStop(trail_pct=0.10)
        # Basis 80 > current 70: underwater, so no fire despite drawdown.
        result = rule.clamp_target("X", 0.10, prices, cost_basis=80.0, high_water_mark=100.0)
        assert result == 0.10


class TestDonchianBreakout:
    """Turtle-style breakout: buy when current close exceeds the prior N-bar high."""

    def test_zero_on_flat_prices(self, flat_prices: PriceHistory) -> None:
        strategy = DonchianBreakout(window=20, breakout_scale=0.02)
        assert strategy.score(flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        # Needs window + 1 bars (N prior + 1 current).
        short = ph(np.full(20, 100.0))
        strategy = DonchianBreakout(window=20)
        assert strategy.score(short) is None

    def test_full_conviction_on_scale_breakout(self) -> None:
        # 20 flat bars at 100, then break to 102 (2% above the prior high).
        prices = ph(np.concatenate([np.full(20, 100.0), np.array([102.0])]))
        strategy = DonchianBreakout(window=20, breakout_scale=0.02)
        score = strategy.score(prices)
        assert score == 1.0

    def test_partial_score_on_partial_breakout(self) -> None:
        # 20 flat bars at 100, then break to 101 — half of the 2% scale.
        prices = ph(np.concatenate([np.full(20, 100.0), np.array([101.0])]))
        strategy = DonchianBreakout(window=20, breakout_scale=0.02)
        score = strategy.score(prices)
        assert score is not None
        assert 0.4 < score < 0.6

    def test_zero_when_below_prior_high(self) -> None:
        # Current close below the prior high — no breakout, no score.
        prices = ph(np.concatenate([np.full(20, 100.0), np.array([99.0])]))
        strategy = DonchianBreakout(window=20, breakout_scale=0.02)
        assert strategy.score(prices) == 0.0

    def test_tie_with_prior_high_does_not_fire(self) -> None:
        # Equalling the prior high is not a "breakout".
        prices = ph(np.concatenate([np.full(20, 100.0), np.array([100.0])]))
        strategy = DonchianBreakout(window=20, breakout_scale=0.02)
        assert strategy.score(prices) == 0.0

    def test_name_and_description(self) -> None:
        strategy = DonchianBreakout(window=20, breakout_scale=0.02)
        assert strategy.name == "DonchianBreakout"
        assert strategy.description

    def test_precompute_matches_score(self) -> None:
        # Vectorized precompute must match the per-bar score() exactly for
        # every prefix — the backtest engine relies on this equivalence.
        rng = np.random.default_rng(seed=11)
        prices = ph(100.0 + np.cumsum(rng.normal(0, 1.0, size=80)))
        strategy = DonchianBreakout(window=20, breakout_scale=0.02)
        precomputed = strategy.precompute(prices)
        assert precomputed is not None
        for i in range(len(prices)):
            expected = strategy.score(prices[: i + 1])
            actual = precomputed[i]
            if expected is None:
                assert np.isnan(actual)
            else:
                assert np.isclose(actual, expected, rtol=1e-9), f"mismatch at i={i}: {actual} vs {expected}"


class TestKeltnerChannel:
    """Breakout entry: bullish when price breaks above EMA + k x ATR upper band."""

    def test_zero_on_flat_prices(self, flat_prices: PriceHistory) -> None:
        strategy = KeltnerChannel(window=20, multiplier=2.0)
        assert strategy.score(flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = ph(np.full(10, 100.0))
        strategy = KeltnerChannel(window=20)
        assert strategy.score(short) is None

    def test_positive_score_on_breakout(self) -> None:
        # Strong uptrend: centerline trails far behind current, ATR rises
        # with the linear ramp — current ends well above the upper band.
        prices = ph(np.linspace(100.0, 140.0, 25))
        strategy = KeltnerChannel(window=20, multiplier=1.0)
        score = strategy.score(prices)
        assert score is not None
        assert score > 0.0

    def test_zero_when_inside_bands(self) -> None:
        # Noise around a flat level — current close barely above the mean,
        # far below the upper band.
        rng = np.random.default_rng(seed=7)
        raw = 100.0 + rng.normal(0, 0.5, size=25)
        raw[-1] = 100.1
        prices = ph(raw)
        strategy = KeltnerChannel(window=20, multiplier=2.0)
        assert strategy.score(prices) == 0.0

    def test_higher_multiplier_suppresses_breakout(self) -> None:
        # Same path, wider band → no longer a breakout.
        prices = ph(np.linspace(100.0, 140.0, 25))
        tight = KeltnerChannel(window=20, multiplier=1.0)
        wide = KeltnerChannel(window=20, multiplier=50.0)
        assert tight.score(prices) > 0.0  # type: ignore[operator]
        assert wide.score(prices) == 0.0

    def test_name_and_description(self) -> None:
        strategy = KeltnerChannel(window=20, multiplier=2.0)
        assert strategy.name == "KeltnerChannel"
        assert strategy.description

    def test_precompute_matches_score(self) -> None:
        rng = np.random.default_rng(seed=13)
        prices = ph(100.0 + np.cumsum(rng.normal(0, 1.2, size=80)))
        strategy = KeltnerChannel(window=20, multiplier=2.0)
        precomputed = strategy.precompute(prices)
        assert precomputed is not None
        for i in range(len(prices)):
            expected = strategy.score(prices[: i + 1])
            actual = precomputed[i]
            if expected is None:
                assert np.isnan(actual)
            else:
                assert np.isclose(actual, expected, rtol=1e-9), f"mismatch at i={i}: {actual} vs {expected}"


class TestChandelierStop:
    """Volatility-adjusted trailing stop: k x ATR below the rolling N-bar high."""

    def test_fires_when_drawdown_exceeds_k_atr(self) -> None:
        # 21 bars rising by $1 (101..121), then a 5-point jump to 125 (peak),
        # then a 15-point crash to 110. Last 22 bars cover everything after bar 0.
        # |diffs| over the window: 20 ones + 4 (one of the ones absorbed) + 15 = ...
        # Concretely: close-to-close ATR ≈ 1.86, highest = 125, stop ≈ 119.4.
        # Current = 110 < 119.4 → fire.
        prices = ph(
            np.concatenate(
                [
                    np.arange(100.0, 121.0, 1.0),  # 21 bars: 100..120
                    np.array([125.0, 110.0]),  # peak then crash
                ]
            )
        )
        rule = ChandelierStop(window=22, multiplier=3.0)
        result = rule.clamp_target("X", 0.10, prices, cost_basis=100.0, high_water_mark=125.0)
        assert result == 0.0

    def test_no_fire_on_steady_rise(self) -> None:
        # All-increasing prices: current == highest, no drawdown at all.
        prices = ph(np.linspace(100.0, 125.0, 25))
        rule = ChandelierStop(window=22, multiplier=3.0)
        result = rule.clamp_target("X", 0.10, prices, cost_basis=100.0, high_water_mark=125.0)
        assert result == 0.10

    def test_fires_even_when_underwater(self) -> None:
        # Chandelier does not gate on cost basis the way TrailingStop does —
        # replacing StopLoss is part of the point. Peak barely clears basis,
        # then price crashes through basis: Chandelier should still fire.
        prices = ph(
            np.concatenate(
                [
                    np.arange(100.0, 121.0, 1.0),  # 21 bars: 100..120
                    np.array([122.0, 95.0]),
                ]
            )
        )
        rule = ChandelierStop(window=22, multiplier=3.0)
        # Basis 120 is above current 95, so position is underwater.
        result = rule.clamp_target("X", 0.10, prices, cost_basis=120.0, high_water_mark=122.0)
        assert result == 0.0

    def test_higher_multiplier_widens_stop(self) -> None:
        # Same price path; raising k should push the stop past the current
        # price and suppress the fire.
        prices = ph(
            np.concatenate(
                [
                    np.arange(100.0, 121.0, 1.0),
                    np.array([125.0, 110.0]),
                ]
            )
        )
        tight = ChandelierStop(window=22, multiplier=3.0)
        wide = ChandelierStop(window=22, multiplier=15.0)
        assert tight.clamp_target("X", 0.10, prices, cost_basis=100.0, high_water_mark=125.0) == 0.0
        assert wide.clamp_target("X", 0.10, prices, cost_basis=100.0, high_water_mark=125.0) == 0.10

    def test_returns_proposed_on_insufficient_history(self) -> None:
        short = ph(np.array([100.0, 101.0, 102.0]))
        rule = ChandelierStop(window=22, multiplier=3.0)
        result = rule.clamp_target("X", 0.10, short, cost_basis=100.0, high_water_mark=102.0)
        assert result == 0.10

    def test_name_and_description(self) -> None:
        rule = ChandelierStop(window=22, multiplier=3.0)
        assert rule.name == "ChandelierStop"
        assert rule.description


class TestParabolicSARExit:
    """Wilder's self-accelerating SAR trailing stop: fire when SAR flips above price."""

    def test_no_clamp_on_steady_rise(self, rising_prices: PriceHistory) -> None:
        rule = ParabolicSARExit()
        result = rule.clamp_target(
            "X",
            0.10,
            rising_prices,
            cost_basis=100.0,
            high_water_mark=float(rising_prices.close.max()),
        )
        assert result == 0.10

    def test_clamps_on_reversal(self, peak_then_drop_prices: PriceHistory) -> None:
        rule = ParabolicSARExit()
        peak = float(peak_then_drop_prices.close.max())
        result = rule.clamp_target(
            "X",
            0.10,
            peak_then_drop_prices,
            cost_basis=100.0,
            high_water_mark=peak,
        )
        assert result == 0.0

    def test_returns_proposed_on_insufficient_history(self) -> None:
        short = ph(np.array([100.0]))
        rule = ParabolicSARExit()
        result = rule.clamp_target(
            "X",
            0.10,
            short,
            cost_basis=100.0,
            high_water_mark=100.0,
        )
        assert result == 0.10

    def test_higher_af_max_flips_faster(self) -> None:
        # A rise then sharp pullback — a faster-accelerating SAR will catch
        # the reversal, while a slow one may not yet have crossed price.
        prices = ph(
            np.concatenate(
                [
                    np.linspace(100.0, 120.0, 15),
                    np.array([119.0, 115.0]),
                ]
            )
        )
        slow = ParabolicSARExit(af_start=0.02, af_step=0.001, af_max=0.005)
        fast = ParabolicSARExit(af_start=0.02, af_step=0.02, af_max=0.20)
        slow_res = slow.clamp_target("X", 0.10, prices, cost_basis=100.0, high_water_mark=120.0)
        fast_res = fast.clamp_target("X", 0.10, prices, cost_basis=100.0, high_water_mark=120.0)
        # Fast SAR must flip on the pullback; slow SAR trailing far below may not.
        assert fast_res == 0.0
        assert slow_res == 0.10

    def test_name_and_description(self) -> None:
        rule = ParabolicSARExit()
        assert rule.name == "ParabolicSARExit"
        assert rule.description


class TestStopLoss:
    def test_clamps_on_loss(self, dropping_prices: PriceHistory) -> None:
        rule = StopLoss(loss_threshold=0.10)
        result = rule.clamp_target("X", 0.10, dropping_prices, cost_basis=100.0, high_water_mark=100.0)
        assert result == 0.0

    def test_no_clamp_above_cost(self, rising_prices: PriceHistory) -> None:
        rule = StopLoss(loss_threshold=0.10)
        result = rule.clamp_target("X", 0.10, rising_prices, cost_basis=100.0, high_water_mark=150.0)
        assert result == 0.10

    def test_no_clamp_without_cost_basis(self, dropping_prices: PriceHistory) -> None:
        rule = StopLoss(loss_threshold=0.10)
        result = rule.clamp_target("X", 0.10, dropping_prices, cost_basis=0.0, high_water_mark=100.0)
        assert result == 0.10

    def test_name_and_description(self) -> None:
        s = StopLoss(loss_threshold=0.05)
        assert s.name == "StopLoss"
        assert s.description


class TestVWAPReversion:
    def test_positive_score_below_average(self, dropping_prices: PriceHistory) -> None:
        strategy = VWAPReversion(window=20, threshold=0.01)
        score = strategy.score(dropping_prices)
        assert score is not None
        assert score > 0.0

    def test_zero_score_above_average(self, rising_prices: PriceHistory) -> None:
        # Buy-only signals clamp at 0 above the band (no negative scores).
        strategy = VWAPReversion(window=20, threshold=0.01)
        score = strategy.score(rising_prices)
        assert score is not None
        assert score == 0.0

    def test_neutral_on_flat(self, flat_prices: PriceHistory) -> None:
        strategy = VWAPReversion(window=20, threshold=0.02)
        assert strategy.score(flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = ph(np.array([100.0] * 5))
        strategy = VWAPReversion(window=20)
        assert strategy.score(short) is None


class TestMovingAverageCrossover:
    def test_positive_score_on_golden_cross(self, ma_crossover_prices: PriceHistory) -> None:
        strategy = MovingAverageCrossover(short_window=10, long_window=30)
        found_signal = False
        for i in range(31, len(ma_crossover_prices)):
            score = strategy.score(ma_crossover_prices[: i + 1])
            if score is not None and score > 0:
                found_signal = True
                break
        assert found_signal, "Expected a positive golden cross score"

    def test_neutral_on_flat(self, flat_prices: PriceHistory) -> None:
        strategy = MovingAverageCrossover(short_window=10, long_window=30)
        assert strategy.score(flat_prices) == 0.0

    def test_insufficient_history(self) -> None:
        short = ph(np.array([100.0] * 20))
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

    def test_parabolic_sar_needs_two_bars(self) -> None:
        assert ParabolicSARExit().warmup_period == 2

    def test_keltner_channel_uses_window(self) -> None:
        assert KeltnerChannel(window=20).warmup_period == 20
        assert KeltnerChannel(window=50).warmup_period == 50

    def test_chandelier_stop_uses_window(self) -> None:
        assert ChandelierStop(window=22).warmup_period == 22
        assert ChandelierStop(window=30).warmup_period == 30

    def test_donchian_breakout_needs_window_plus_one(self) -> None:
        # Donchian compares current to the prior N bars, so it needs
        # N + 1 bars total before it can score.
        assert DonchianBreakout(window=20).warmup_period == 21
        assert DonchianBreakout(window=55).warmup_period == 56

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
