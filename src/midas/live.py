"""Live analysis engine — polls prices and emits alerts in real time."""

from __future__ import annotations

import time
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from midas.allocator import Allocator
from midas.data.provider import DataProvider
from midas.models import (
    AllocationConstraints,
    Direction,
    MechanicalIntent,
    Order,
    PortfolioConfig,
)
from midas.output import print_alert, print_status
from midas.rebalancer import Rebalancer
from midas.restrictions import RestrictionTracker
from midas.strategies.base import Strategy


class LiveEngine:
    def __init__(
        self,
        portfolio: PortfolioConfig,
        allocator: Allocator,
        rebalancer: Rebalancer,
        provider: DataProvider,
        mechanical_strategies: list[Strategy] | None = None,
        constraints: AllocationConstraints | None = None,
        poll_interval: int = 60,
        dry_run: bool = False,
        history_days: int = 60,
    ) -> None:
        self._portfolio = portfolio
        self._allocator = allocator
        self._rebalancer = rebalancer
        self._mechanical = mechanical_strategies or []
        self._constraints = constraints or AllocationConstraints()
        self._provider = provider
        self._poll_interval = poll_interval
        self._dry_run = dry_run
        self._history_days = history_days
        # Track (ticker, direction, shares) from last tick to suppress duplicate alerts
        self._last_order_keys: set[tuple[str, Direction, float]] = set()
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

        # Build context (cost_basis from portfolio config)
        context: dict[str, dict[str, object]] = {}
        current_prices: dict[str, float] = {}
        for ticker in tickers:
            if ticker in price_data and len(price_data[ticker]) > 0:
                current_prices[ticker] = float(price_data[ticker].iloc[-1])
                holding = self._portfolio.get_holding(ticker)
                ctx: dict[str, object] = {}
                if holding and holding.cost_basis:
                    ctx["cost_basis"] = holding.cost_basis
                context[ticker] = ctx

        active_tickers = [t for t in tickers if t in price_data]

        # Phase 1-3: Allocate
        allocation = self._allocator.allocate(active_tickers, price_data, context)

        # Phase 4: Rebalance
        positions = {}
        for t in active_tickers:
            holding = self._portfolio.get_holding(t)
            positions[t] = holding.shares if holding else 0.0

        rebalance_orders = self._rebalancer.generate_orders(
            allocation,
            positions,
            current_prices,
            self._portfolio.available_cash,
            self._constraints,
        )

        # Phase 5: Mechanical
        mechanical_intents: list[MechanicalIntent] = []
        for strat in self._mechanical:
            for ticker in active_tickers:
                if ticker in price_data:
                    ticker_ctx = context.get(ticker, {})
                    intents = strat.generate_intents(
                        ticker,
                        price_data[ticker],
                        **ticker_ctx,
                    )
                    mechanical_intents.extend(intents)

        sell_proceeds = sum(o.estimated_value for o in rebalance_orders if o.direction == Direction.SELL)
        buy_cost = sum(o.estimated_value for o in rebalance_orders if o.direction == Direction.BUY)
        post_cash = self._portfolio.available_cash + sell_proceeds - buy_cost
        mechanical_orders = self._rebalancer.size_mechanical(
            mechanical_intents,
            post_cash,
            current_prices,
        )

        all_orders = rebalance_orders + mechanical_orders

        # Filter restricted orders
        filtered: list[Order] = []
        for order in all_orders:
            if self._restriction_tracker and self._restriction_tracker.is_blocked(
                order.ticker,
                order.direction,
                today,
            ):
                continue
            filtered.append(order)

        # Emit alerts only when the order set changes
        current_keys = {(o.ticker, o.direction, o.shares) for o in filtered if o.shares > 0}
        if current_keys == self._last_order_keys:
            return
        self._last_order_keys = current_keys

        now = datetime.now(tz=UTC)
        remaining_cash = self._portfolio.available_cash
        for order in filtered:
            if order.shares <= 0:
                continue
            if order.direction == Direction.BUY:
                remaining_cash -= order.estimated_value
            else:
                remaining_cash += order.estimated_value
            print_alert(order, remaining_cash, now, dry_run=self._dry_run)
