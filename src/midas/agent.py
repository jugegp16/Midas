"""Agent — ties a strategy to a portfolio and manages cooldowns."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from midas.models import Direction, PortfolioConfig, Signal
from midas.strategies.base import Strategy

PriceArray = pd.Series | np.ndarray


class Agent:
    def __init__(
        self,
        strategy: Strategy,
        cooldown_days: int = 5,
        tickers: list[str] | None = None,
    ) -> None:
        self.strategy = strategy
        self._cooldown_days = cooldown_days
        self._tickers = tickers
        # Cooldown tracker: (ticker, direction) -> last signal date
        self._last_signal: dict[tuple[str, Direction], date] = {}

    def run(
        self,
        portfolio: PortfolioConfig,
        price_data: dict[str, PriceArray],
        today: date | None = None,
        cost_basis_overrides: dict[str, float] | None = None,
    ) -> list[Signal]:
        """Run the strategy against all applicable tickers. Returns new signals.

        Args:
            cost_basis_overrides: If provided, these per-ticker cost basis
                values take priority over the portfolio config. Used by the
                backtest engine to track simulated weighted-average cost basis.
        """
        signals: list[Signal] = []
        tickers = self._tickers or [h.ticker for h in portfolio.holdings]

        for ticker in tickers:
            if ticker not in price_data:
                continue

            if cost_basis_overrides and ticker in cost_basis_overrides:
                cost_basis: float | None = cost_basis_overrides[ticker]
            else:
                holding = portfolio.get_holding(ticker)
                cost_basis = holding.cost_basis if holding else None

            raw_signals = self.strategy.evaluate(
                ticker, price_data[ticker], cost_basis=cost_basis
            )

            for sig in raw_signals:
                if self._is_cooled_down(sig, today):
                    signals.append(sig)
                    signal_date = today or sig.timestamp.date()
                    self._last_signal[(sig.ticker, sig.direction)] = signal_date

        return signals

    def _is_cooled_down(self, signal: Signal, today: date | None) -> bool:
        key = (signal.ticker, signal.direction)
        last = self._last_signal.get(key)
        if last is None:
            return True
        current = today or signal.timestamp.date()
        return (current - last).days >= self._cooldown_days
