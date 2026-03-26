"""Live analysis engine — polls prices and emits signals in real time."""

from __future__ import annotations

import time
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from midas.agent import Agent
from midas.data.provider import DataProvider
from midas.models import Direction, Order, PortfolioConfig
from midas.output import print_alert, print_circuit_breaker_alert, print_status
from midas.restrictions import RestrictionTracker
from midas.sizing import SizingEngine


class LiveEngine:
    def __init__(
        self,
        portfolio: PortfolioConfig,
        agents: list[Agent],
        provider: DataProvider,
        sizing_engine: SizingEngine | None = None,
        poll_interval: int = 60,
        dry_run: bool = False,
        history_days: int = 60,
    ) -> None:
        self._portfolio = portfolio
        self._agents = agents
        self._provider = provider
        self._sizing = sizing_engine or SizingEngine()
        self._poll_interval = poll_interval
        self._dry_run = dry_run
        self._history_days = history_days
        self._daily_deployed = 0.0
        self._last_reset_date: date | None = None
        self._restriction_tracker: RestrictionTracker | None = None
        if portfolio.trading_restrictions:
            self._restriction_tracker = RestrictionTracker(
                portfolio.trading_restrictions,
            )

    def run(self) -> None:
        tickers = [h.ticker for h in self._portfolio.holdings]
        print_status(
            f"Starting {'dry run' if self._dry_run else 'live'} analysis "
            f"for {len(tickers)} tickers, polling every {self._poll_interval}s"
        )

        try:
            while True:
                self._tick(tickers)
                time.sleep(self._poll_interval)
        except KeyboardInterrupt:
            print_status("Stopped.")

    def _tick(self, tickers: list[str]) -> None:
        today = date.today()

        # Reset daily deployed on new day
        if self._last_reset_date != today:
            self._daily_deployed = 0.0
            self._last_reset_date = today

        # Fetch recent history for all tickers
        end = today
        start = end - timedelta(days=self._history_days)
        price_data: dict[str, pd.Series] = {}
        for ticker in tickers:
            try:
                price_data[ticker] = self._provider.get_history(ticker, start, end)
            except Exception as e:
                print_status(f"Warning: failed to fetch {ticker}: {e}")

        if not price_data:
            return

        # Run agents
        orders: list[Order] = []
        for agent in self._agents:
            signals = agent.run(self._portfolio, price_data, today=today)
            for signal in signals:
                holding = self._portfolio.get_holding(signal.ticker)
                position_shares = holding.shares if holding else 0
                order = self._sizing.size_order(
                    signal=signal,
                    available_cash=self._portfolio.available_cash,
                    position_shares=position_shares,
                    cash_infusion=self._portfolio.cash_infusion,
                    daily_deployed=self._daily_deployed,
                    today=today,
                )
                if order is not None:
                    if self._restriction_tracker and self._restriction_tracker.is_blocked(
                        order.ticker, order.direction, today,
                    ):
                        continue
                    orders.append(order)
                    if self._restriction_tracker and order.shares > 0:
                        self._restriction_tracker.record_trade(
                            order.ticker, order.direction, today,
                        )
                    if order.shares > 0 and order.direction == Direction.BUY:
                        self._daily_deployed += order.estimated_value

        # Emit alerts
        now = datetime.now(tz=UTC)
        for order in orders:
            if order.shares == 0:
                print_circuit_breaker_alert(order, dry_run=self._dry_run)
            else:
                remaining_cash = self._portfolio.available_cash - (
                    order.estimated_value if order.direction == Direction.BUY else 0
                )
                print_alert(
                    order, remaining_cash, now, dry_run=self._dry_run
                )
