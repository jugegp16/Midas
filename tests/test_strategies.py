"""Tests for individual strategies."""

import pandas as pd

from midas.models import Direction
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
        signals = strategy.evaluate("FLAT", flat_prices)
        assert signals == []

    def test_buy_signal_on_drop(self, dropping_prices: pd.Series) -> None:
        strategy = MeanReversion(window=30, threshold=0.05)
        signals = strategy.evaluate("DROP", dropping_prices)
        assert len(signals) == 1
        assert signals[0].direction == Direction.BUY
        assert signals[0].ticker == "DROP"
        assert 0.0 <= signals[0].strength <= 1.0

    def test_insufficient_history(self) -> None:
        short = pd.Series([100.0] * 5, name="SHORT")
        strategy = MeanReversion(window=30)
        assert strategy.evaluate("SHORT", short) == []

    def test_name_and_description(self) -> None:
        s = MeanReversion(window=20, threshold=0.08)
        assert "20" in s.name
        assert "0.08" in s.name
        assert len(s.suitability) > 0


class TestProfitTaking:
    def test_sell_signal_on_gain(self, rising_prices: pd.Series) -> None:
        strategy = ProfitTaking(gain_threshold=0.15)
        signals = strategy.evaluate("RISE", rising_prices, cost_basis=100.0)
        assert len(signals) == 1
        assert signals[0].direction == Direction.SELL

    def test_no_signal_without_cost_basis(self, rising_prices: pd.Series) -> None:
        strategy = ProfitTaking(gain_threshold=0.15)
        assert strategy.evaluate("RISE", rising_prices) == []
        assert strategy.evaluate("RISE", rising_prices, cost_basis=None) == []

    def test_no_signal_below_threshold(self, flat_prices: pd.Series) -> None:
        strategy = ProfitTaking(gain_threshold=0.20)
        signals = strategy.evaluate("FLAT", flat_prices, cost_basis=100.0)
        assert signals == []


class TestMomentum:
    def test_buy_on_crossover(self, crossover_prices: pd.Series) -> None:
        strategy = Momentum(window=20)
        # Evaluate at each point — should find a crossover
        found_signal = False
        for i in range(21, len(crossover_prices)):
            signals = strategy.evaluate("CROSS", crossover_prices.iloc[: i + 1])
            if signals:
                assert signals[0].direction == Direction.BUY
                found_signal = True
                break
        assert found_signal, "Expected a momentum crossover signal"

    def test_no_signal_on_flat(self, flat_prices: pd.Series) -> None:
        strategy = Momentum(window=20)
        signals = strategy.evaluate("FLAT", flat_prices)
        assert signals == []

    def test_insufficient_history(self) -> None:
        short = pd.Series([100.0] * 10, name="SHORT")
        strategy = Momentum(window=20)
        assert strategy.evaluate("SHORT", short) == []


class TestRSIOversold:
    def test_buy_signal_on_oversold(
        self, volatile_dropping_prices: pd.Series
    ) -> None:
        strategy = RSIOversold(window=14, oversold_threshold=40.0)
        signals = strategy.evaluate("VDROP", volatile_dropping_prices)
        assert len(signals) == 1
        assert signals[0].direction == Direction.BUY
        assert 0.0 <= signals[0].strength <= 1.0

    def test_no_signal_on_flat(self, flat_prices: pd.Series) -> None:
        strategy = RSIOversold(window=14, oversold_threshold=30.0)
        signals = strategy.evaluate("FLAT", flat_prices)
        assert signals == []

    def test_insufficient_history(self) -> None:
        short = pd.Series([100.0] * 5)
        strategy = RSIOversold(window=14)
        assert strategy.evaluate("SHORT", short) == []

    def test_name_and_description(self) -> None:
        s = RSIOversold(window=10, oversold_threshold=25.0)
        assert "10" in s.name
        assert "25" in s.name
        assert len(s.suitability) > 0
        assert s.description


class TestRSIOverbought:
    def test_sell_signal_on_overbought(
        self, volatile_rising_prices: pd.Series
    ) -> None:
        strategy = RSIOverbought(window=14, overbought_threshold=65.0)
        signals = strategy.evaluate("VRISE", volatile_rising_prices)
        assert len(signals) == 1
        assert signals[0].direction == Direction.SELL

    def test_no_signal_on_flat(self, flat_prices: pd.Series) -> None:
        strategy = RSIOverbought(window=14, overbought_threshold=70.0)
        signals = strategy.evaluate("FLAT", flat_prices)
        assert signals == []

    def test_insufficient_history(self) -> None:
        short = pd.Series([100.0] * 5)
        strategy = RSIOverbought(window=14)
        assert strategy.evaluate("SHORT", short) == []


class TestBollingerBand:
    def test_buy_on_lower_band(self, dropping_prices: pd.Series) -> None:
        strategy = BollingerBand(window=20, num_std=1.5)
        signals = strategy.evaluate("DROP", dropping_prices)
        assert len(signals) == 1
        assert signals[0].direction == Direction.BUY

    def test_no_signal_on_flat(self, flat_prices: pd.Series) -> None:
        strategy = BollingerBand(window=20, num_std=2.0)
        signals = strategy.evaluate("FLAT", flat_prices)
        assert signals == []

    def test_insufficient_history(self) -> None:
        short = pd.Series([100.0] * 5)
        strategy = BollingerBand(window=20)
        assert strategy.evaluate("SHORT", short) == []

    def test_name_and_description(self) -> None:
        s = BollingerBand(window=20, num_std=2.5)
        assert "20" in s.name
        assert "2.5" in s.name


class TestMACDCrossover:
    def test_buy_on_crossover(self, crossover_prices: pd.Series) -> None:
        strategy = MACDCrossover(fast_period=12, slow_period=26, signal_period=9)
        # Walk through price history to find crossover
        found_signal = False
        for i in range(35, len(crossover_prices)):
            signals = strategy.evaluate("CROSS", crossover_prices.iloc[: i + 1])
            if signals:
                assert signals[0].direction == Direction.BUY
                found_signal = True
                break
        assert found_signal, "Expected a MACD crossover signal"

    def test_no_signal_on_flat(self, flat_prices: pd.Series) -> None:
        strategy = MACDCrossover()
        signals = strategy.evaluate("FLAT", flat_prices)
        assert signals == []

    def test_insufficient_history(self) -> None:
        short = pd.Series([100.0] * 10)
        strategy = MACDCrossover()
        assert strategy.evaluate("SHORT", short) == []


class TestDollarCostAveraging:
    def test_signal_on_frequency(self, flat_prices: pd.Series) -> None:
        strategy = DollarCostAveraging(frequency_days=10)
        # Feed exactly 10 days — should trigger
        signals = strategy.evaluate("FLAT", flat_prices.iloc[:10])
        assert len(signals) == 1
        assert signals[0].direction == Direction.BUY
        assert signals[0].strength == 0.5

    def test_no_signal_off_frequency(self, flat_prices: pd.Series) -> None:
        strategy = DollarCostAveraging(frequency_days=10)
        signals = strategy.evaluate("FLAT", flat_prices.iloc[:11])
        assert signals == []

    def test_insufficient_history(self) -> None:
        short = pd.Series([100.0] * 3)
        strategy = DollarCostAveraging(frequency_days=5)
        assert strategy.evaluate("SHORT", short) == []

    def test_name_and_description(self) -> None:
        s = DollarCostAveraging(frequency_days=7)
        assert "7" in s.name
        assert s.description


class TestGapDownRecovery:
    def test_buy_on_gap_recovery(
        self, gap_down_recovery_prices: pd.Series
    ) -> None:
        strategy = GapDownRecovery(gap_threshold=0.03)
        # Walk to find a gap-down + recovery
        found_signal = False
        for i in range(3, len(gap_down_recovery_prices)):
            signals = strategy.evaluate(
                "GAP", gap_down_recovery_prices.iloc[: i + 1]
            )
            if signals:
                assert signals[0].direction == Direction.BUY
                found_signal = True
                break
        assert found_signal, "Expected a gap-down recovery signal"

    def test_no_signal_on_flat(self, flat_prices: pd.Series) -> None:
        strategy = GapDownRecovery(gap_threshold=0.03)
        signals = strategy.evaluate("FLAT", flat_prices)
        assert signals == []

    def test_insufficient_history(self) -> None:
        short = pd.Series([100.0, 95.0])
        strategy = GapDownRecovery()
        assert strategy.evaluate("SHORT", short) == []


class TestTrailingStop:
    def test_sell_on_drawdown(self, peak_then_drop_prices: pd.Series) -> None:
        strategy = TrailingStop(trail_pct=0.08)
        signals = strategy.evaluate(
            "PEAK", peak_then_drop_prices, cost_basis=100.0
        )
        assert len(signals) == 1
        assert signals[0].direction == Direction.SELL

    def test_no_signal_without_cost_basis(
        self, peak_then_drop_prices: pd.Series
    ) -> None:
        strategy = TrailingStop(trail_pct=0.08)
        assert strategy.evaluate("PEAK", peak_then_drop_prices) == []

    def test_no_signal_on_rising(self, rising_prices: pd.Series) -> None:
        strategy = TrailingStop(trail_pct=0.10)
        signals = strategy.evaluate("RISE", rising_prices, cost_basis=100.0)
        assert signals == []

    def test_name_and_description(self) -> None:
        s = TrailingStop(trail_pct=0.15)
        assert "0.15" in s.name
        assert s.description


class TestStopLoss:
    def test_sell_on_loss(self, dropping_prices: pd.Series) -> None:
        strategy = StopLoss(loss_threshold=0.10)
        signals = strategy.evaluate("DROP", dropping_prices, cost_basis=100.0)
        assert len(signals) == 1
        assert signals[0].direction == Direction.SELL

    def test_no_signal_without_cost_basis(
        self, dropping_prices: pd.Series
    ) -> None:
        strategy = StopLoss(loss_threshold=0.10)
        assert strategy.evaluate("DROP", dropping_prices) == []

    def test_no_signal_above_cost(self, rising_prices: pd.Series) -> None:
        strategy = StopLoss(loss_threshold=0.10)
        signals = strategy.evaluate("RISE", rising_prices, cost_basis=100.0)
        assert signals == []

    def test_name_and_description(self) -> None:
        s = StopLoss(loss_threshold=0.05)
        assert "0.05" in s.name
        assert s.description


class TestVWAPReversion:
    def test_buy_below_average(self, dropping_prices: pd.Series) -> None:
        strategy = VWAPReversion(window=20, threshold=0.01)
        signals = strategy.evaluate("DROP", dropping_prices)
        assert len(signals) == 1
        assert signals[0].direction == Direction.BUY

    def test_sell_above_average(self, rising_prices: pd.Series) -> None:
        strategy = VWAPReversion(window=20, threshold=0.01)
        signals = strategy.evaluate("RISE", rising_prices)
        assert len(signals) == 1
        assert signals[0].direction == Direction.SELL

    def test_no_signal_on_flat(self, flat_prices: pd.Series) -> None:
        strategy = VWAPReversion(window=20, threshold=0.02)
        signals = strategy.evaluate("FLAT", flat_prices)
        assert signals == []

    def test_insufficient_history(self) -> None:
        short = pd.Series([100.0] * 5)
        strategy = VWAPReversion(window=20)
        assert strategy.evaluate("SHORT", short) == []


class TestMovingAverageCrossover:
    def test_buy_on_golden_cross(
        self, ma_crossover_prices: pd.Series
    ) -> None:
        strategy = MovingAverageCrossover(short_window=10, long_window=30)
        found_signal = False
        for i in range(31, len(ma_crossover_prices)):
            signals = strategy.evaluate(
                "MACROSS", ma_crossover_prices.iloc[: i + 1]
            )
            if signals and signals[0].direction == Direction.BUY:
                found_signal = True
                break
        assert found_signal, "Expected a golden cross signal"

    def test_no_signal_on_flat(self, flat_prices: pd.Series) -> None:
        strategy = MovingAverageCrossover(short_window=10, long_window=30)
        signals = strategy.evaluate("FLAT", flat_prices)
        assert signals == []

    def test_insufficient_history(self) -> None:
        short = pd.Series([100.0] * 20)
        strategy = MovingAverageCrossover(short_window=10, long_window=50)
        assert strategy.evaluate("SHORT", short) == []

    def test_name_and_description(self) -> None:
        s = MovingAverageCrossover(short_window=15, long_window=45)
        assert "15" in s.name
        assert "45" in s.name
