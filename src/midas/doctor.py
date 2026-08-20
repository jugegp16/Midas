"""Setup health checks for `midas doctor`.

Runs stepped diagnostics per configured integration — Discord today,
broker credentials later — and reports each capability as pass/fail
with a concrete fix hint. The philosophy is the inverse of the live
engine's: live swallows delivery errors to protect the tick loop,
doctor surfaces them loudly because the errors are the product.

Nothing here mutates live state: no pending orders, no tally counts,
no report anchor. Test messages are unmistakably labeled and edited to
"test complete" so nobody trades off one.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from midas.alerts import (
    CONFIRM_EMOJIS,
    DECLINE_EMOJIS,
    DailyReport,
    DiscordApiError,
    DiscordBotClient,
    order_embed,
    report_embed,
)
from midas.data.provider import DataProvider
from midas.live_state import load_state
from midas.market_hours import MARKET_TZ
from midas.models import AlertsConfig, Direction, Order, OrderContext

REACTION_WAIT_SECONDS = 60.0
REACTION_POLL_SECONDS = 3.0
PRICE_LOOKBACK_DAYS = 7

Echo = Callable[[str], None]


@dataclass
class DoctorResult:
    """Aggregate outcome of a doctor run."""

    checks: int = 0
    failures: int = 0
    warnings: int = 0

    @property
    def healthy(self) -> bool:
        return self.failures == 0


class _Reporter:
    """Check-line rendering with pass/fail/warn bookkeeping."""

    def __init__(self, echo: Echo, result: DoctorResult) -> None:
        self._echo = echo
        self._result = result

    def ok(self, message: str) -> None:
        self._result.checks += 1
        self._echo(f"  ✓ {message}")

    def fail(self, message: str, hint: str) -> None:
        self._result.checks += 1
        self._result.failures += 1
        self._echo(f"  ✗ {message}")
        self._echo(f"    fix: {hint}")

    def warn(self, message: str) -> None:
        self._result.warnings += 1
        self._echo(f"  ! {message}")

    def note(self, message: str) -> None:
        self._echo(f"  - {message}")


def _hint_for(exc: DiscordApiError, subject: str) -> str:
    """Map an API failure to the most likely operator fix."""
    if exc.status == 401:
        return "the bot token is invalid — reset it in the developer portal and re-export MIDAS_DISCORD_BOT_TOKEN"
    if exc.status == 403:
        return f"the bot lacks access to {subject} — invite it to the server / grant it visibility of the channel"
    if exc.status == 404:
        return f"{subject} was not found — double-check the channel id (developer mode → Copy Channel ID)"
    return "network or Discord availability problem — retry; if it persists, check connectivity"


def _test_order() -> Order:
    return Order(
        ticker="TEST",
        direction=Direction.BUY,
        shares=1.0,
        price=0.01,
        estimated_value=0.01,
        context=OrderContext(
            contributions={},
            blended_score=0.0,
            target_weight=0.0,
            current_weight=0.0,
            reason="connectivity test — DO NOT TRADE",
            source="midas doctor",
        ),
    )


def _check_channel(client: DiscordBotClient, checker: _Reporter, label: str, channel_id: str) -> bool:
    try:
        channel = client.get_channel(channel_id)
    except DiscordApiError as exc:
        checker.fail(f"{label} channel {channel_id}: {exc}", _hint_for(exc, f"the {label} channel"))
        return False
    checker.ok(f"{label} channel visible (#{channel.get('name', channel_id)})")
    return True


def _post_probe(
    client: DiscordBotClient,
    checker: _Reporter,
    label: str,
    channel_id: str,
    payload: dict[str, object],
) -> str | None:
    """Post a test message, read it back, and return its id (or None)."""
    try:
        message_id = client.create_message(channel_id, payload)
    except DiscordApiError as exc:
        checker.fail(f"{label}: post failed ({exc})", _hint_for(exc, f"the {label} channel"))
        return None
    checker.ok(f"{label}: test message posted")
    try:
        client.get_message(channel_id, message_id)
    except DiscordApiError as exc:
        checker.fail(
            f"{label}: could not read the message back ({exc}) — confirm-mode polling would be blind",
            "grant the bot the Read Message History permission",
        )
        return message_id
    checker.ok(f"{label}: message read back (Read Message History OK)")
    return message_id


def _wait_for_reaction(client: DiscordBotClient, checker: _Reporter, channel_id: str, message_id: str) -> None:
    checker.note(f"waiting up to {REACTION_WAIT_SECONDS:.0f}s — react ✅ or ❌ to the test message now")
    deadline = time.monotonic() + REACTION_WAIT_SECONDS
    while time.monotonic() < deadline:
        try:
            message = client.get_message(channel_id, message_id)
        except DiscordApiError as exc:
            checker.fail(f"reaction poll failed ({exc})", _hint_for(exc, "the intra-day channel"))
            return
        for reaction in message.get("reactions") or []:
            name = str((reaction.get("emoji") or {}).get("name") or "").replace("\ufe0f", "")
            if name in CONFIRM_EMOJIS or name in DECLINE_EMOJIS:
                checker.ok(f"reaction round-trip verified ({name} seen)")
                return
        time.sleep(REACTION_POLL_SECONDS)
    checker.warn("no reaction within the wait window — round-trip not verified (transport itself is fine)")


def _snapshot_report(state_path: Path, provider: DataProvider) -> tuple[DailyReport, str]:
    """Build an end-of-day report from real state, or labeled sample data.

    Returns:
        ``(report, provenance)`` where provenance describes the data
        source for the operator.
    """
    now = datetime.now(tz=UTC)
    session_date = now.astimezone(MARKET_TZ).date()
    try:
        state = load_state(state_path)
        prices: dict[str, float] = {}
        for ticker, lots in state.lots.items():
            if sum(lot.shares for lot in lots) <= 0:
                continue
            frame = provider.get_history(
                ticker,
                session_date - timedelta(days=PRICE_LOOKBACK_DAYS),
                session_date,
            )
            prices[ticker] = float(frame["close"].iloc[-1])
        equity = state.available_cash + sum(
            sum(lot.shares for lot in lots) * prices[ticker] for ticker, lots in state.lots.items() if ticker in prices
        )
        report = DailyReport(
            session_date=session_date,
            equity=equity,
            previous_equity=state.last_report_equity,
            cash=state.available_cash,
            positions=sum(1 for lots in state.lots.values() if sum(lot.shares for lot in lots) > 0),
            pending=len(state.pending_orders),
            alerts_posted=state.tally_posted,
            confirmed=state.tally_confirmed,
            declined=state.tally_declined,
            expired=state.tally_expired,
            cppi_scale=1.0,
            vol_target_scale=1.0,
        )
        return report, "real state snapshot"
    except Exception as exc:
        # Any data problem (missing state, provider failure, malformed
        # file) degrades to sample numbers — the transport test must not
        # fail because yfinance hiccuped.
        report = DailyReport(
            session_date=session_date,
            equity=100_000.0,
            previous_equity=99_000.0,
            cash=5_000.0,
            positions=4,
            pending=1,
            alerts_posted=3,
            confirmed=2,
            declined=1,
            expired=0,
            cppi_scale=1.0,
            vol_target_scale=1.0,
        )
        return report, f"sample numbers (state/prices unavailable: {exc})"


def run_discord_checks(
    alerts_config: AlertsConfig | None,
    bot_token: str,
    state_path: Path,
    provider: DataProvider,
    echo: Echo,
    wait_reaction: bool = False,
) -> DoctorResult:
    """Run the Discord section of `midas doctor`.

    Args:
        alerts_config: The portfolio's ``alerts:`` block, if any.
        bot_token: Resolved ``MIDAS_DISCORD_BOT_TOKEN`` value ("" = unset).
        state_path: Live state file, for the real-snapshot report.
        provider: Price source for marking the snapshot to market.
        echo: Line sink for check output.
        wait_reaction: Block for an operator reaction on the intra-day
            test message to verify the confirm round trip.

    Returns:
        Aggregate result; ``healthy`` is False when any check failed.
    """
    result = DoctorResult()
    checker = _Reporter(echo, result)
    echo("Discord")

    intra_day = alerts_config.discord_intra_day_channel_id.strip() if alerts_config else ""
    end_of_day = alerts_config.discord_end_of_day_channel_id.strip() if alerts_config else ""
    token = bot_token.strip()

    if not token and not intra_day and not end_of_day:
        checker.note("not configured (no channels, no bot token) — nothing to check")
        return result
    if not token:
        checker.fail(
            "channel id configured but MIDAS_DISCORD_BOT_TOKEN is not set",
            "export the bot token (developer portal → Bot → Reset Token)",
        )
        return result
    if not intra_day and not end_of_day:
        checker.fail(
            "MIDAS_DISCORD_BOT_TOKEN is set but no channel id is configured",
            "add discord_intra_day_channel_id and/or discord_end_of_day_channel_id to the alerts: block",
        )
        return result

    client = DiscordBotClient(token, timeout_seconds=alerts_config.timeout_seconds if alerts_config else 5.0)

    try:
        me = client.get_current_user()
    except DiscordApiError as exc:
        checker.fail(f"bot token rejected ({exc})", _hint_for(exc, "the bot"))
        return result
    checker.ok(f"bot token valid (logged in as {me.get('username', '?')})")

    now = datetime.now(tz=UTC)
    if intra_day and _check_channel(client, checker, "intra-day", intra_day):
        embed = order_embed(_test_order(), now)
        embed["title"] = "🧪 TEST — not a real order"
        message_id = _post_probe(
            client, checker, "intra-day", intra_day, {"embeds": [embed], "content": "🧪 Connectivity test"}
        )
        if message_id is not None:
            if wait_reaction:
                _wait_for_reaction(client, checker, intra_day, message_id)
            try:
                client.edit_message(intra_day, message_id, {"content": "🧪 Test complete — connectivity check only"})
                checker.ok("intra-day: message edit OK (booked/declined annotations will work)")
            except DiscordApiError as exc:
                checker.fail(f"intra-day: edit failed ({exc})", _hint_for(exc, "the intra-day channel"))

    if end_of_day and _check_channel(client, checker, "end-of-day", end_of_day):
        report, provenance = _snapshot_report(state_path, provider)
        checker.note(f"report payload: {provenance}")
        embed = report_embed(report)
        embed["title"] = f"🧪 TEST report — {report.session_date.isoformat()}"
        _post_probe(client, checker, "end-of-day", end_of_day, {"embeds": [embed]})

    return result
