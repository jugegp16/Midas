"""Live analysis engine — polls prices and emits alerts in real time."""

from __future__ import annotations

import errno
import logging
import os
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]

from midas.alerts import DailyReport, DiscordReporter, OrderConfirmer, report_embed
from midas.allocator import AllocationResult, Allocator
from midas.data.price_history import PriceHistory
from midas.data.provider import DataProvider
from midas.live_state import (
    IntentCooldown,
    LiveState,
    PendingOrder,
    PurchaseDate,
    SellBreakdown,
    aggregate_cost_basis,
    apply_buy,
    apply_sell,
    load_or_seed,
    resolve_purchase_date,
    save_atomic,
)
from midas.market_hours import MARKET_TZ, is_market_open, next_session_open, session_close
from midas.models import (
    DEFAULT_PENDING_TTL_HOURS,
    DEFAULT_REALERT_HOURS,
    AllocationConstraints,
    Direction,
    HoldingPeriod,
    Order,
    PortfolioConfig,
    TradeRecord,
)
from midas.order_sizer import OrderSizer
from midas.output import print_alert, print_status
from midas.restrictions import RestrictionTracker
from midas.strategies.base import ExitRule, max_warmup, warmup_bars_to_calendar_days
from midas.trade_log import append_trade

logger = logging.getLogger(__name__)

# A pending order suppresses re-alerts for the same (ticker, direction)
# unless the recomputed share count moves by more than this fraction AND
# the pending alert has outlived the re-alert cooldown — then the stale
# alert is expired and replaced. (Direction flips replace immediately.)
PENDING_RESIZE_TOLERANCE = 0.10

# The close-of-day report fires on the first tick at or after this many
# minutes before the session close (~15:55 ET with the default 5).
REPORT_MINUTES_BEFORE_CLOSE = 5.0
# Off-hours sleep is chunked so a suspended laptop or clock jump can't
# oversleep a session by more than one chunk.
MAX_SLEEP_CHUNK_SECONDS = 3600.0


def _acquire_state_lock(lock_path: Path) -> int:
    """Take an exclusive non-blocking advisory lock on *lock_path*.

    Two ``midas live`` processes against the same state file would otherwise
    both load, both compute, both write — ``os.replace`` picks a winner and
    the loser's fills are lost silently. This converts that silent corruption
    into a loud error. The lock is held for the lifetime of the process; flock
    releases on fd close, which Python handles automatically at interpreter
    exit.

    Args:
        lock_path: Sidecar lockfile path next to the state file.

    Returns:
        The open file descriptor holding the lock.

    Raises:
        RuntimeError: If another process already holds the lock.
        OSError: On any other locking failure.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(lock_fd)
        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
            msg = (
                f"another midas live process appears to hold the state lock at {lock_path}; "
                f"refusing to start. If you're sure no other process is running, remove the lock file."
            )
            raise RuntimeError(msg) from exc
        raise
    return lock_fd


class LiveEngine:
    """Polls prices, runs the allocation pipeline, and emits order alerts.

    Holds an exclusive lock on a sidecar lockfile for its lifetime so only
    one process mutates the persisted ``LiveState`` at a time.
    """

    def __init__(
        self,
        portfolio: PortfolioConfig,
        allocator: Allocator,
        order_sizer: OrderSizer,
        provider: DataProvider,
        state_path: Path,
        exit_rules: list[ExitRule] | None = None,
        constraints: AllocationConstraints | None = None,
        poll_interval: int = 60,
        dry_run: bool = False,
        history_days: int | None = None,
        confirmer: OrderConfirmer | None = None,
        pending_ttl_hours: float = DEFAULT_PENDING_TTL_HOURS,
        realert_hours: float = DEFAULT_REALERT_HOURS,
        reporter: DiscordReporter | None = None,
        market_hours_only: bool = True,
    ) -> None:
        """Acquire the state lock, then load or seed persistent state.

        Args:
            portfolio: Portfolio configuration (tickers, strategies, infusions).
            allocator: Signal-blending allocator producing target weights.
            order_sizer: Converts target weights into sized orders.
            provider: Price-history data source.
            state_path: Path of the persisted live-state YAML.
            exit_rules: Optional exit rules that clamp targets downward.
            constraints: Allocation constraints; defaults to no constraints.
            poll_interval: Seconds between ticks.
            dry_run: When True, alerts are labeled as dry-run and confirm
                mode is disabled (assumed fills, no Discord posts).
            history_days: Explicit history window override (tests); derived
                from strategy warmups when None.
            confirmer: When set (and not dry-run), enables confirm mode:
                orders become pending at alert time and fill only on
                operator confirmation. None keeps assumed-fill behavior.
            pending_ttl_hours: Hours a pending order stays confirmable.
            realert_hours: Minimum hours before the same (ticker,
                direction) is re-alerted after a confirmed fill or a
                resize of a still-pending alert.
            reporter: Discord destination for the close-of-day report;
                None keeps the report terminal-only.
            market_hours_only: When True (default), tick only during
                regular US equity sessions and sleep until the next open
                otherwise. False restores 24/7 polling (debugging).

        Raises:
            RuntimeError: If file locking is unavailable on this platform or
                another live process already holds the state lock.
        """
        if fcntl is None:
            msg = (
                "LiveEngine requires fcntl-based file locking, which is unavailable "
                "on this platform (Windows). The live engine is currently POSIX-only."
            )
            raise RuntimeError(msg)
        self._state_path = state_path
        self._trade_log_path = state_path.with_suffix(state_path.suffix + ".trades.csv")
        # Take the lock on a sidecar lockfile before touching the state.
        self._lock_path = state_path.with_suffix(state_path.suffix + ".lock")
        self._lock_fd: int | None = _acquire_state_lock(self._lock_path)

        # Once the lock is held, any failure during the rest of __init__ must
        # release it; otherwise a supervisor (or REPL) that catches the
        # exception and retries would see a misleading "another midas live
        # process" RuntimeError on the second attempt within the same process.
        # ``BaseException`` is intentional — keyboard interrupts during
        # ``load_or_seed`` should also release the lock.
        try:
            self._state: LiveState = load_or_seed(portfolio, state_path)
            self._portfolio = portfolio
            self._allocator = allocator
            self._order_sizer = order_sizer
            self._exit_rules = exit_rules or []
            self._constraints = constraints or AllocationConstraints()
            self._provider = provider
            self._poll_interval = poll_interval
            self._dry_run = dry_run
            if dry_run and confirmer is not None:
                logger.info("dry run: confirm mode disabled, alerts stay terminal-only with assumed fills")
                confirmer = None
            self._confirmer = confirmer
            if self._confirmer is None and self._state.pending_orders:
                # Stranded pendings are a double-book vector: this mode
                # assume-fills the same recomputed intents, and a later
                # return to confirm mode would replay a stale reaction on
                # top of that at the old alert price.
                stranded = ", ".join(
                    f"{p.direction.value} {p.ticker} ({p.shares:g} sh, message {p.message_id})"
                    for p in self._state.pending_orders
                )
                logger.warning(
                    "confirm mode is OFF but the state file holds %d pending order(s) "
                    "awaiting confirmation: %s. They will not be polled — operator "
                    "reactions are ignored, and this mode's assumed fills can "
                    "double-book the same intent when confirm mode returns. Restart "
                    "with the Discord bot configured (and without --dry-run), or "
                    "remove pending_orders from the state file by hand.",
                    len(self._state.pending_orders),
                    stranded,
                )
            self._pending_ttl = timedelta(hours=pending_ttl_hours)
            self._realert_cooldown = timedelta(hours=realert_hours)
            self._reporter = reporter
            self._market_hours_only = market_hours_only
            self._last_equity: float | None = None
            self._last_risk_telemetry: tuple[float, float] = (1.0, 1.0)
            # In-memory per-session alert counters for the report. Reset
            # after each report; an intra-day restart undercounts, which
            # the report tolerates (state-derived figures do not).
            self._alerts_posted = 0
            self._alerts_confirmed = 0
            self._alerts_declined = 0
            self._alerts_expired = 0
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
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Release the state lockfile.

        In production the engine lives for the lifetime of the process and
        the lock is released implicitly at interpreter exit when the file
        descriptor is closed. Tests that construct multiple ``LiveEngine``s
        against the same state file (e.g. the backtest/live parity suite)
        must call this between instances or the second construction will
        race the first's GC and fail with ``RuntimeError`` on the
        ``LOCK_EX | LOCK_NB`` acquire. Idempotent.
        """
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_fd)
                self._lock_fd = None

    def __enter__(self) -> LiveEngine:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def run(self) -> None:
        """Poll during market sessions, ticking every ``poll_interval`` seconds until Ctrl-C."""
        tickers = [holding.ticker for holding in self._portfolio.holdings]
        print_status(
            f"Starting {'dry run' if self._dry_run else 'live'} analysis "
            f"for {len(tickers)} tickers, polling every {self._poll_interval}s"
        )

        try:
            while True:
                now = datetime.now(tz=UTC)
                if self._market_hours_only and not is_market_open(now):
                    self._sleep_until_open(now)
                    continue
                try:
                    self._tick(tickers)
                    self._maybe_send_report(datetime.now(tz=UTC))
                except Exception:
                    # A live process watching real money must not die
                    # overnight on a one-tick bug — state transitions are
                    # individually durable, so skipping a tick is safe.
                    # Same crash-proofing rationale as _fetch_histories.
                    logger.exception("tick failed; continuing next poll")
                time.sleep(self._poll_interval)
        except KeyboardInterrupt:
            print_status("Stopped.")

    @staticmethod
    def _sleep_until_open(now: datetime) -> None:
        """Sleep (in chunks) until the next session open.

        Chunked so a laptop suspend or clock jump re-evaluates the target
        at most ``MAX_SLEEP_CHUNK_SECONDS`` late rather than oversleeping
        into an entire missed session.
        """
        next_open = next_session_open(now)
        local = next_open.astimezone(MARKET_TZ)
        print_status(f"Market closed — sleeping until {local:%Y-%m-%d %H:%M} ET")
        while True:
            remaining = (next_open - datetime.now(tz=UTC)).total_seconds()
            if remaining <= 0:
                return
            time.sleep(min(remaining, MAX_SLEEP_CHUNK_SECONDS))

    def _maybe_send_report(self, now: datetime) -> None:
        """Emit the close-of-day report once per session, near the close."""
        if not self._market_hours_only:
            return
        session_date = now.astimezone(MARKET_TZ).date()
        if self._state.last_report_date == session_date:
            return
        seconds_to_close = (session_close(now) - now).total_seconds()
        if seconds_to_close > REPORT_MINUTES_BEFORE_CLOSE * 60 or seconds_to_close < 0:
            return
        self._send_report(session_date)

    def _send_report(self, session_date: date) -> None:
        """Assemble and deliver the report; persist the equity anchor."""
        equity = self._last_equity if self._last_equity is not None else self._state.available_cash
        report = DailyReport(
            session_date=session_date,
            equity=equity,
            previous_equity=self._state.last_report_equity,
            cash=self._state.available_cash,
            positions=sum(1 for lots in self._state.lots.values() if sum(lot.shares for lot in lots) > 0),
            pending=len(self._state.pending_orders),
            alerts_posted=self._alerts_posted,
            confirmed=self._alerts_confirmed,
            declined=self._alerts_declined,
            expired=self._alerts_expired,
            cppi_scale=self._last_risk_telemetry[0],
            vol_target_scale=self._last_risk_telemetry[1],
        )
        # Persist the anchor BEFORE delivery: a Discord failure must not
        # re-arm the report and double-post on a later tick, and the
        # day-over-day delta must survive restarts.
        self._state.last_report_date = session_date
        self._state.last_report_equity = equity
        save_atomic(self._state, self._state_path)

        embed = report_embed(report)
        print_status(f"Close of day — {session_date.isoformat()}")
        for field_entry in embed["fields"]:
            print_status(f"  {field_entry['name']}: {field_entry['value']}")
        if self._reporter is not None:
            self._reporter.post_report(report)
        self._alerts_posted = 0
        self._alerts_confirmed = 0
        self._alerts_declined = 0
        self._alerts_expired = 0

    def _append_log_row(
        self,
        record: TradeRecord,
        *,
        cost_basis: float | None,
        purchase_date: PurchaseDate,
    ) -> None:
        """Append one trade-log row, salvaging the data on IO failure.

        State is already durable when this runs, so a failed append is never
        retried by a later tick — print the row instead so the operator can
        hand-add it to the (hand-editable) log.
        """
        try:
            append_trade(self._trade_log_path, record, cost_basis=cost_basis, purchase_date=purchase_date)
        except OSError as exc:
            logger.error(
                "failed to append to trade log %s (%s); add this row by hand: "
                "date=%s ticker=%s direction=%s shares=%s price=%s strategy=%s "
                "holding_period=%s purchase_date=%s cost_basis=%s",
                self._trade_log_path,
                exc,
                record.date.isoformat(),
                record.ticker,
                record.direction.value,
                record.shares,
                record.price,
                record.strategy_name,
                record.holding_period.value if record.holding_period else "",
                purchase_date,
                cost_basis,
            )

    def _fetch_histories(
        self,
        tickers: list[str],
        today: date,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, PriceHistory]]:
        """Fetch recent history for all tickers and convert to PriceHistory.

        Both calls share a try/except so a misshapen frame (missing OHLCV
        columns, bad index) skips that ticker for the tick instead of
        crashing the entire poll loop.
        """
        start = today - timedelta(days=self._history_days)
        price_data: dict[str, pd.DataFrame] = {}
        price_history: dict[str, PriceHistory] = {}
        for ticker in tickers:
            try:
                df = self._provider.get_history(ticker, start, today)
                price_history[ticker] = PriceHistory.from_dataframe(df)
                price_data[ticker] = df
            except Exception as exc:
                print_status(f"Warning: failed to fetch {ticker}: {exc}")
        return price_data, price_history

    def _positions(self, tickers: list[str]) -> dict[str, float]:
        """Per-ticker share counts derived from state lots, not the YAML."""
        return {ticker: sum(lot.shares for lot in self._state.lots.get(ticker, [])) for ticker in tickers}

    def _portfolio_value(self, positions: dict[str, float], current_prices: dict[str, float]) -> float:
        """Available cash plus mark-to-market value of *positions*."""
        return self._state.available_cash + sum(shares * current_prices[ticker] for ticker, shares in positions.items())

    def _advance_high_water_marks(self, active_tickers: list[str], current_prices: dict[str, float]) -> None:
        """Advance per-ticker HWM in state to the latest close (never regress).

        Only held tickers count: HWM on a not-yet-bought ticker would seed
        ``TrailingStop`` against a pre-purchase peak on the very first held
        day, silently more conservative than backtest, which only tracks
        HWM on positions with shares > 0.
        """
        held_tickers = {ticker for ticker, lots in self._state.lots.items() if sum(lot.shares for lot in lots) > 0}
        for ticker in active_tickers:
            if ticker not in held_tickers:
                continue
            close = current_prices[ticker]
            self._state.high_water_marks[ticker] = max(self._state.high_water_marks.get(ticker, close), close)

    def _advance_cash_infusion(self, today: date) -> None:
        """Credit the cash infusion if due — BEFORE allocator/sizer decisions.

        This makes the new cash available for today's allocations (matches
        backtest's _run_day, which credits the infusion at the start of the
        day).
        """
        infusion = self._portfolio.cash_infusion
        if (
            infusion is not None
            and self._state.cash_infusion_next_date is not None
            and today >= self._state.cash_infusion_next_date
        ):
            self._state.available_cash += infusion.amount
            # CashInfusion.advance() mutates next_date in place; align it with state, advance, copy back.
            infusion.next_date = self._state.cash_infusion_next_date
            infusion.advance()
            self._state.cash_infusion_next_date = infusion.next_date

    def _clamp_targets(
        self,
        allocation: AllocationResult,
        active_tickers: list[str],
        positions: dict[str, float],
        price_history: dict[str, PriceHistory],
        current_prices: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, tuple[str, str]]]:
        """Let exit rules clamp proposed targets downward (LEAN pattern)."""
        clamped_targets = dict(allocation.targets)
        clamp_attribution: dict[str, tuple[str, str]] = {}
        for rule in self._exit_rules:
            for ticker in active_tickers:
                if positions.get(ticker, 0.0) <= 0:
                    continue
                proposed = clamped_targets.get(ticker, 0.0)
                if proposed <= 0:
                    continue
                cost_basis = aggregate_cost_basis(self._state.lots.get(ticker, []))
                hwm = self._state.high_water_marks.get(ticker, current_prices[ticker])
                clamped = rule.clamp_target(ticker, proposed, price_history[ticker], cost_basis, hwm)
                if clamped < proposed:
                    clamped_targets[ticker] = clamped
                    if ticker not in clamp_attribution:
                        reason = rule.clamp_reason(ticker, price_history[ticker], cost_basis, hwm)
                        clamp_attribution[ticker] = (rule.name, reason)
        return clamped_targets, clamp_attribution

    def _filter_restricted(self, orders: list[Order], today: date) -> list[Order]:
        """Drop orders blocked by trading restrictions (no-op when untracked)."""
        if self._restriction_tracker is None:
            return orders
        return [
            order for order in orders if not self._restriction_tracker.is_blocked(order.ticker, order.direction, today)
        ]

    def _apply_fills(self, orders: list[Order], today: date) -> dict[int, SellBreakdown]:
        """Apply assumed fills to the in-memory state.

        Captures per-SELL breakdowns so the trade-log append has shares/basis/
        dates per ST/LT bucket. Order is a frozen dataclass but contains an
        OrderContext with a dict field (contributions), making it unhashable.
        Use id(order) — safe because *orders* is a stable list across both
        passes within _tick.
        """
        sell_breakdowns: dict[int, SellBreakdown] = {}
        for order in orders:
            if order.shares <= 0:
                continue
            if order.direction == Direction.BUY:
                apply_buy(self._state, order.ticker, order.shares, order.price, today)
            else:
                sell_breakdowns[id(order)] = apply_sell(self._state, order.ticker, order.shares, order.price, today)
            if self._restriction_tracker is not None:
                self._restriction_tracker.record_trade(order.ticker, order.direction, today)
        return sell_breakdowns

    def _log_fills(self, orders: list[Order], sell_breakdowns: dict[int, SellBreakdown], today: date) -> None:
        """Append a row per BUY and per non-empty ST/LT SELL bucket to the trade log.

        State is durable first, and a failed append is NOT retried on later
        ticks: positions re-derive from state, so the deltas are gone. On
        failure, _append_log_row prints the row so the operator can add it
        by hand (the log is hand-editable by design).
        """
        for order in orders:
            if order.shares <= 0:
                continue
            if order.direction == Direction.BUY:
                buy_record = TradeRecord(
                    date=today,
                    ticker=order.ticker,
                    direction=Direction.BUY,
                    shares=order.shares,
                    price=order.price,
                    strategy_name=order.context.source,
                    purchase_date=today,
                )
                self._append_log_row(buy_record, cost_basis=None, purchase_date=today)
                continue
            breakdown = sell_breakdowns[id(order)]
            self._log_sell_buckets(order.ticker, order.price, order.context.source, breakdown, today)

    def _emit_alerts(self, orders: list[Order], pre_fill_cash: float) -> None:
        """Emit alerts only when the order set changes since the last tick."""
        current_keys = {(order.ticker, order.direction, order.shares) for order in orders if order.shares > 0}
        if current_keys == self._last_order_keys:
            return
        self._last_order_keys = current_keys

        now = datetime.now(tz=UTC)
        remaining_cash = pre_fill_cash
        for order in orders:
            if order.shares <= 0:
                continue
            if order.direction == Direction.BUY:
                remaining_cash -= order.estimated_value
            else:
                remaining_cash += order.estimated_value
            print_alert(order, remaining_cash, now, dry_run=self._dry_run)

    def _resolve_pending(self, today: date, now: datetime) -> None:
        """Poll operator decisions on pending orders; apply, decline, or expire.

        Runs at the top of the tick so a confirmed fill updates positions
        and cash BEFORE this tick's allocation decisions.

        Durability ordering: for every transition, the pending order is
        removed (and a confirmed fill applied) in state and persisted via
        ``save_atomic`` BEFORE any other side effect — trade-log rows, the
        restriction clock, terminal output, Discord edits. The operator's
        ✅ reaction is durable external state that is re-polled after a
        restart; if the booking were not on disk first, a crash mid-tick
        would replay the confirmation and double-book the fill.
        """
        assert self._confirmer is not None
        self._state.intent_cooldowns = [cooldown for cooldown in self._state.intent_cooldowns if now < cooldown.until]
        for pending in list(self._state.pending_orders):
            decision = self._confirmer.poll_decision(pending.message_id)
            if decision == "unavailable":
                # The poll itself failed — the operator may have reacted
                # and executed. Expiring here would drop a real fill, so
                # the order waits (even past TTL) until Discord answers.
                continue
            expired = decision is None and now - pending.created_at > self._pending_ttl
            if decision is None and not expired:
                continue

            self._state.pending_orders = [p for p in self._state.pending_orders if p is not pending]
            booked: SellBreakdown | None | Literal["buy", "nothing-held"] = None
            note = f"{pending.shares:g} sh @ ${pending.price:,.2f} on {today.isoformat()}"
            # Cooldowns are recorded in the same durable transition as the
            # removal, so suppression survives a crash/restart like a fill.
            # A decline silences the intent for the rest of the confirmation
            # window ("no for today"); a confirmed fill only for the shorter
            # re-alert cooldown (scale-ins at most hourly, not every tick).
            if decision == "confirmed":
                booked, note = self._book_confirmed_fill(pending, today, note)
                self._state.intent_cooldowns.append(
                    IntentCooldown(
                        ticker=pending.ticker,
                        direction=pending.direction,
                        until=now + self._realert_cooldown,
                    )
                )
            elif decision == "declined":
                self._state.intent_cooldowns.append(
                    IntentCooldown(ticker=pending.ticker, direction=pending.direction, until=now + self._pending_ttl)
                )
            save_atomic(self._state, self._state_path)

            # State is durable — everything below is replay-safe salvage.
            if decision == "confirmed":
                self._alerts_confirmed += 1
                self._finish_confirmed_fill(pending, booked, note, today)
            elif decision == "declined":
                self._alerts_declined += 1
                print_status(f"Declined: {pending.direction.value} {pending.ticker} ({pending.shares:g} shares)")
                self._confirmer.mark_declined(pending.message_id)
            else:
                self._alerts_expired += 1
                print_status(
                    f"Expired unconfirmed: {pending.direction.value} {pending.ticker} ({pending.shares:g} shares)"
                )
                self._confirmer.mark_expired(pending.message_id)

    def _book_confirmed_fill(
        self,
        pending: PendingOrder,
        today: date,
        note: str,
    ) -> tuple[SellBreakdown | Literal["buy", "nothing-held"], str]:
        """Apply a confirmed fill to in-memory state (no IO, no side effects).

        Fills at the alert price — the same price assumption the legacy
        path made; confirmation changes only the WHETHER (broker sync will later
        correct the at-what from broker truth).

        Returns:
            ``(booked, note)`` where *booked* is ``"buy"``, the sell's
            ``SellBreakdown``, or ``"nothing-held"`` when a SELL found no
            shares left; *note* is the (possibly clamp-annotated) summary
            used for logging and the Discord annotation.
        """
        if pending.direction == Direction.BUY:
            apply_buy(self._state, pending.ticker, pending.shares, pending.price, today)
            if self._state.available_cash < 0:
                logger.warning(
                    "available cash went negative (%.2f) booking confirmed BUY %s — "
                    "verify against the brokerage balance",
                    self._state.available_cash,
                    pending.ticker,
                )
            return "buy", note
        held = sum(lot.shares for lot in self._state.lots.get(pending.ticker, []))
        # Position may have shrunk since the alert (e.g. another sell
        # confirmed first). Clamp rather than let apply_sell's oversell
        # assertion kill the poll loop.
        shares = min(pending.shares, held)
        if shares <= 0:
            return "nothing-held", f"{note} — no shares held in state"
        if shares < pending.shares:
            logger.warning(
                "confirmed SELL %s clamped from %s to %s shares (position shrank since alert)",
                pending.ticker,
                pending.shares,
                shares,
            )
            note = f"{shares:g} sh @ ${pending.price:,.2f} on {today.isoformat()} (clamped from {pending.shares:g})"
        return apply_sell(self._state, pending.ticker, shares, pending.price, today), note

    def _finish_confirmed_fill(
        self,
        pending: PendingOrder,
        booked: SellBreakdown | None | Literal["buy", "nothing-held"],
        note: str,
        today: date,
    ) -> None:
        """Post-persistence side effects of a confirmed fill: log, clock, annotate."""
        assert self._confirmer is not None
        if booked == "nothing-held":
            logger.error(
                "confirmed SELL %s for %s shares but none are held in state; nothing booked",
                pending.ticker,
                pending.shares,
            )
            self._confirmer.mark_unfillable(pending.message_id, note)
            return
        if booked == "buy":
            record = TradeRecord(
                date=today,
                ticker=pending.ticker,
                direction=Direction.BUY,
                shares=pending.shares,
                price=pending.price,
                strategy_name=pending.strategy_name,
                purchase_date=today,
            )
            self._append_log_row(record, cost_basis=None, purchase_date=today)
        else:
            assert isinstance(booked, SellBreakdown)
            self._log_sell_buckets(pending.ticker, pending.price, pending.strategy_name, booked, today)
        if self._restriction_tracker is not None:
            # Round-trip clocks start at real execution, not at alert time.
            self._restriction_tracker.record_trade(pending.ticker, pending.direction, today)
        print_status(f"Booked: {pending.direction.value} {pending.ticker} — {note}")
        self._confirmer.mark_booked(pending.message_id, note)

    def _log_sell_buckets(
        self,
        ticker: str,
        price: float,
        strategy_name: str,
        breakdown: SellBreakdown,
        today: date,
    ) -> None:
        """Append one trade-log row per non-empty ST/LT bucket of a sell."""
        buckets = (
            (HoldingPeriod.SHORT_TERM, breakdown.st_shares, breakdown.st_basis, breakdown.st_purchase_dates),
            (HoldingPeriod.LONG_TERM, breakdown.lt_shares, breakdown.lt_basis, breakdown.lt_purchase_dates),
        )
        for period, shares, basis, dates in buckets:
            if shares <= 0:
                continue
            purchase = resolve_purchase_date(dates)
            record = TradeRecord(
                date=today,
                ticker=ticker,
                direction=Direction.SELL,
                shares=shares,
                price=price,
                strategy_name=strategy_name,
                holding_period=period,
                purchase_date=purchase,
            )
            self._append_log_row(record, cost_basis=basis, purchase_date=purchase)

    def _post_pending_alerts(self, orders: list[Order], now: datetime, pre_fill_cash: float) -> None:
        """Post new orders for confirmation, honoring the pending gate.

        A live pending order suppresses re-alerts for the same
        (ticker, direction). A direction flip supersedes the stale alert
        immediately (an exit signal must not wait), but a mere resize
        beyond ``PENDING_RESIZE_TOLERANCE`` only supersedes once the
        pending alert is older than the re-alert cooldown — live prices
        wobble every tick, and expire-and-replace churn on every size
        recompute buried the operator in "Expired" spam. Intents under an
        active cooldown (operator declined, or a fill just booked) are
        skipped outright; cooldowns are pruned in ``_resolve_pending``.

        A pending is re-polled immediately before being superseded: a
        reaction that landed after the top-of-tick poll keeps it alive
        for ``_resolve_pending`` instead of being silently dropped.
        """
        assert self._confirmer is not None
        cooled_keys = {(cooldown.ticker, cooldown.direction) for cooldown in self._state.intent_cooldowns}
        remaining_cash = pre_fill_cash
        for order in orders:
            if order.shares <= 0 or (order.ticker, order.direction) in cooled_keys:
                continue
            keep_pending: list[PendingOrder] = []
            superseded: list[tuple[PendingOrder, Order]] = []
            suppressed = False
            for pending in self._state.pending_orders:
                if pending.ticker != order.ticker:
                    keep_pending.append(pending)
                    continue
                resized = (
                    pending.direction == order.direction
                    and abs(order.shares - pending.shares) > PENDING_RESIZE_TOLERANCE * pending.shares
                )
                replace = pending.direction != order.direction or (
                    resized and now - pending.created_at >= self._realert_cooldown
                )
                # Re-poll immediately before superseding: a reaction may have
                # landed after _resolve_pending's top-of-tick poll, and
                # replacing the pending now would silently drop a fill the
                # operator already executed at the broker. Any decision — or
                # an unanswerable poll — keeps the pending; _resolve_pending
                # owns those transitions next tick.
                if replace and self._confirmer.poll_decision(pending.message_id) is not None:
                    replace = False
                if replace:
                    superseded.append((pending, order))
                else:
                    suppressed = True
                    keep_pending.append(pending)
            self._state.pending_orders = keep_pending
            if superseded:
                # Persist the removal BEFORE annotating Discord — the same
                # persist-before-side-effects rule as _resolve_pending.
                save_atomic(self._state, self._state_path)
                for stale, replacement in superseded:
                    print_status(
                        f"Superseded pending {stale.direction.value} {stale.ticker}: "
                        f"recomputed as {replacement.direction.value} {replacement.shares:g} shares"
                    )
                    self._confirmer.mark_superseded(stale.message_id)
            if suppressed:
                continue

            # Terminal stays the channel of record; the projected-cash
            # subtotal mirrors _emit_alerts.
            if order.direction == Direction.BUY:
                remaining_cash -= order.estimated_value
            else:
                remaining_cash += order.estimated_value
            print_alert(order, remaining_cash, now, dry_run=self._dry_run)
            print_status("Awaiting confirmation in Discord (✅ books the fill, ❌ skips it)")

            message_id = self._confirmer.post_order(order, now)
            if message_id is None:
                # Not persisted as pending: the order is re-proposed and the
                # post retried on a later tick.
                continue
            self._alerts_posted += 1
            self._state.pending_orders.append(
                PendingOrder(
                    ticker=order.ticker,
                    direction=order.direction,
                    shares=order.shares,
                    price=order.price,
                    strategy_name=order.context.source,
                    reason=order.context.reason,
                    created_at=now,
                    message_id=message_id,
                )
            )
            # Persist immediately: the Discord message is durable external
            # state the moment it posts. A crash before the tick-end save
            # would orphan it — the operator's reaction on a message no
            # restart knows about would be silently ignored, and the next
            # tick would post a duplicate. Same reasoning as the
            # persist-before-side-effects rule in _resolve_pending.
            save_atomic(self._state, self._state_path)

    def _tick(self, tickers: list[str]) -> None:
        """Run one poll cycle: fetch, allocate, size, fill, persist, alert."""
        today = date.today()

        price_data, price_history = self._fetch_histories(tickers, today)
        if not price_data:
            return

        # If any held position is missing from price_data, we can't compute an
        # accurate portfolio denominator for current_weights — skip the tick
        # rather than let Option A hold inflated weights based on partial info.
        missing_held = [
            holding.ticker
            for holding in self._portfolio.holdings
            if holding.shares > 0 and holding.ticker not in price_data
        ]
        if missing_held:
            print_status(f"Skipping tick: missing price data for held positions {missing_held}. Will retry next poll.")
            return

        current_prices: dict[str, float] = {}
        for ticker in tickers:
            if ticker in price_data and len(price_data[ticker]) > 0:
                current_prices[ticker] = float(price_data[ticker]["close"].iloc[-1])

        active_tickers = [ticker for ticker in tickers if ticker in price_data]

        self._advance_high_water_marks(active_tickers, current_prices)
        self._advance_cash_infusion(today)

        now = datetime.now(tz=UTC)
        if self._confirmer is not None:
            # Resolve operator decisions FIRST so a confirmed fill updates
            # positions and cash before this tick's allocation decisions.
            self._resolve_pending(today, now)

        # Current positions + weights (weights feed Option A: neutral=hold).
        positions = self._positions(active_tickers)

        # Pass None (not {}) when the denominator is zero so the allocator
        # falls back to its equal-weight baseline.
        total_value = self._portfolio_value(positions, current_prices)
        current_weights: dict[str, float] | None = None
        if total_value > 0:
            current_weights = {
                ticker: (positions[ticker] * current_prices[ticker]) / total_value for ticker in active_tickers
            }

        # Phase 1: Allocator scores entry signals and blends to target weights.
        # ``current_drawdown`` feeds the optional CPPI overlay (Phase 4a).
        # ``peak_equity`` is now persistent across runs via the state
        # sidecar — without this argument the overlay would always be a
        # no-op in live (its raison d'être for this PR).
        peak = self._state.peak_equity
        current_drawdown = (peak - total_value) / peak if peak is not None and peak > 0 and total_value < peak else 0.0
        allocation = self._allocator.allocate(
            active_tickers,
            price_history,
            current_weights=current_weights,
            current_drawdown=current_drawdown,
        )
        telemetry = allocation.risk_telemetry
        self._last_risk_telemetry = (telemetry.cppi_scale, telemetry.vol_target_scale)

        # Phase 2: Exit rules clamp proposed targets downward.
        clamped_targets, clamp_attribution = self._clamp_targets(
            allocation, active_tickers, positions, price_history, current_prices
        )

        # Size sells and filter restriction-blocked sells *before* computing
        # post-sell cash. Otherwise a blocked sell would leak phantom proceeds
        # into the buy pass, sizing buys against cash that will never arrive.
        exit_orders = self._order_sizer.size_sells(
            clamped_targets,
            positions,
            current_prices,
            total_value,
            clamp_attribution,
        )
        exit_orders = self._filter_restricted(exit_orders, today)
        sell_proceeds = sum(order.estimated_value for order in exit_orders)
        post_sell_cash = self._state.available_cash + sell_proceeds
        if self._confirmer is not None:
            # Cash already promised to unconfirmed BUY alerts is spoken
            # for — sizing against it would let two confirmed buys drive
            # cash negative with no guard.
            reserved = sum(
                pending.shares * pending.price
                for pending in self._state.pending_orders
                if pending.direction == Direction.BUY
            )
            post_sell_cash = max(0.0, post_sell_cash - reserved)

        clamped_allocation = AllocationResult(
            targets=clamped_targets,
            contributions=allocation.contributions,
            blended_scores=allocation.blended_scores,
        )

        buy_orders = self._order_sizer.size_buys(
            clamped_allocation,
            positions,
            current_prices,
            post_sell_cash,
            self._constraints,
            total_value=total_value,
        )
        buy_orders = self._filter_restricted(buy_orders, today)

        filtered = exit_orders + buy_orders

        # Capture the pre-fill cash baseline for the alert-emission loop below.
        # apply_buy/apply_sell mutate state.available_cash, so seeding the
        # running subtotal from state after fills would double-count.
        pre_fill_cash = self._state.available_cash

        if self._confirmer is not None:
            # Confirm mode: nothing fills at alert time. New orders are
            # posted for confirmation; fills happened in _resolve_pending
            # (already reflected in positions/cash above).
            self._post_pending_alerts(filtered, now, pre_fill_cash)
            current_equity = self._portfolio_value(self._positions(active_tickers), current_prices)
            self._last_equity = current_equity
            self._state.peak_equity = max(self._state.peak_equity or 0.0, current_equity)
            # Persist pending-order transitions along with HWM/peak/infusion.
            save_atomic(self._state, self._state_path)
            return

        sell_breakdowns = self._apply_fills(filtered, today)

        # Update peak equity from the current portfolio value (post-fills).
        current_equity = self._portfolio_value(self._positions(active_tickers), current_prices)
        self._last_equity = current_equity
        self._state.peak_equity = max(self._state.peak_equity or 0.0, current_equity)

        # Persist state at the end of the tick (HWM/peak/infusion always advance,
        # even on no-change ticks where alert printing is suppressed below).
        save_atomic(self._state, self._state_path)

        self._log_fills(filtered, sell_breakdowns, today)
        self._emit_alerts(filtered, pre_fill_cash)
