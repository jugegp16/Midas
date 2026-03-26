"""Tests for individual strategies."""

import pandas as pd

from midas.models import Direction
from midas.strategies.mean_reversion import MeanReversion
from midas.strategies.momentum import Momentum
from midas.strategies.profit_taking import ProfitTaking


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
