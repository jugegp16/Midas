"""Live analysis engine — polls prices and emits alerts in real time."""

from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from midas.allocator import Allocator
from midas.data.provider import DataProvider
from midas.models import (
    AllocationConstraints,
    Direction,
    ExitIntent,
    PortfolioConfig,
    PositionLot,
)
from midas.order_sizer import OrderSizer
from midas.output import print_alert, print_status
from midas.restrictions import RestrictionTracker
from midas.strategies.base import ExitRule, max_warmup, warmup_bars_to_calendar_days

logger = logging.getLogger(__name__)


class LiveEngine:
    def __init__(
        self,
        portfolio: PortfolioConfig,
        allocator: Allocator,
        order_sizer: OrderSizer,
        provider: DataProvider,
        exit_rules: list[ExitRule] | None = None,
        constraints: AllocationConstraints | None = None,
        poll_interval: int = 60,
        dry_run: bool = False,
        history_days: int | None = None,
    ) -> None:
        self._portfolio = portfolio
        self._allocator = allocator
        self._order_sizer = order_sizer
        self._exit_rules = exit_rules or []
        self._constraints = constraints or AllocationConstraints()
        self._provider = provider
        self._poll_interval = poll_interval
        self._dry_run = dry_run
        # Derive the history window from the largest warmup required across
        # configured strategies (plus slack for weekends/holidays). An explicit
        # ``history_days`` override is still honored for tests.
        if history_days is not None:
            self._history_days = history_days
        else:
            warmup_bars = max_warmup([*allocator.strategies, *self._exit_rules])
            self._history_days = warmup_bars_to_calendar_days(warmup_bars)
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

        # If any held position is missing from price_data, we can't compute an
        # accurate portfolio denominator for current_weights — skip the tick
        # rather than let Option A hold inflated weights based on partial info.
        missing_held = [h.ticker for h in self._portfolio.holdings if h.shares > 0 and h.ticker not in price_data]
        if missing_held:
            print_status(f"Skipping tick: missing price data for held positions {missing_held}. Will retry next poll.")
            return

        current_prices: dict[str, float] = {}
        for ticker in tickers:
            if ticker in price_data and len(price_data[ticker]) > 0:
                current_prices[ticker] = float(price_data[ticker].iloc[-1])

        active_tickers = [t for t in tickers if t in price_data]

        # Convert pd.Series to numpy arrays at the boundary — strategies
        # and the allocator operate on np.ndarray for performance.
        price_arrays: dict[str, np.ndarray] = {t: np.asarray(price_data[t]) for t in active_tickers}

        # Current positions + weights (weights feed Option A: neutral=hold).
        positions = {}
        for t in active_tickers:
            holding = self._portfolio.get_holding(t)
            positions[t] = holding.shares if holding else 0.0

        # Pass None (not {}) when the denominator is zero so the allocator
        # falls back to its equal-weight baseline.
        total_value = self._portfolio.available_cash + sum(positions[t] * current_prices[t] for t in active_tickers)
        current_weights: dict[str, float] | None = None
        if total_value > 0:
            current_weights = {t: (positions[t] * current_prices[t]) / total_value for t in active_tickers}

        # Phase 1: Allocator scores entry signals and blends to target weights.
        allocation = self._allocator.allocate(
            active_tickers,
            price_arrays,
            current_weights=current_weights,
        )

        # Phase 2: Exit rules. Live mode has no per-lot history (the portfolio
        # config records only an aggregate cost_basis), so each held position
        # is presented as a single synthetic lot. This is good enough for
        # stop-loss / trailing-stop / profit-taking which all read from
        # current price vs cost basis. A future iteration should track lots
        # across ticks for true FIFO accounting (see follow-up issue).
        exit_intents: list[ExitIntent] = []
        for rule in self._exit_rules:
            for ticker in active_tickers:
                if positions.get(ticker, 0.0) <= 0:
                    continue
                holding = self._portfolio.get_holding(ticker)
                if holding is None:
                    continue
                if holding.cost_basis is None:
                    # No basis on file — fall back to today's price so the lot
                    # is "fresh". This silently disables stop-loss / profit-
                    # taking on the position (gain == 0%, loss == 0%) until a
                    # real basis is recorded; warn loudly so the operator
                    # knows their stops aren't actually armed.
                    logger.warning(
                        "%s: no cost_basis in portfolio config — using current "
                        "price as fallback. Stop-loss and profit-taking exits "
                        "are effectively disabled for this ticker until a real "
                        "basis is recorded.",
                        ticker,
                    )
                    cost_basis = current_prices[ticker]
                else:
                    cost_basis = holding.cost_basis
                # Synthesize a single lot with HWM = max(basis, current price)
                # so trailing-stop has a sensible reference until proper
                # per-lot tracking is added.
                hwm = max(cost_basis, current_prices[ticker])
                lots = [
                    PositionLot(
                        shares=holding.shares,
                        purchase_date=None,
                        cost_basis=cost_basis,
                        high_water_mark=hwm,
                    )
                ]
                exit_intents.extend(rule.evaluate_exit(ticker, lots, price_arrays[ticker]))

        # Size sells and filter restriction-blocked sells *before* computing
        # post-sell cash. Otherwise a blocked sell would leak phantom proceeds
        # into the buy pass, sizing buys against cash that will never arrive.
        exit_orders = self._order_sizer.size_exits(exit_intents, positions, current_prices)
        if self._restriction_tracker:
            exit_orders = [
                o for o in exit_orders if not self._restriction_tracker.is_blocked(o.ticker, o.direction, today)
            ]
        sell_proceeds = sum(o.estimated_value for o in exit_orders)
        post_sell_cash = self._portfolio.available_cash + sell_proceeds

        buy_orders = self._order_sizer.size_buys(
            allocation,
            positions,
            current_prices,
            post_sell_cash,
            self._constraints,
            total_value=total_value,
        )
        if self._restriction_tracker:
            buy_orders = [
                o for o in buy_orders if not self._restriction_tracker.is_blocked(o.ticker, o.direction, today)
            ]

        filtered = exit_orders + buy_orders

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
