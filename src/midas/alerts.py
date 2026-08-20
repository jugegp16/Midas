"""Discord delivery and fill confirmation for live alerts.

The terminal remains the alert channel of record — Discord is additive.
A delivery or polling failure must never break a live tick: every network
error is logged and swallowed at this module's public surface, mirroring
the trade-log salvage philosophy in ``live.py``.

Delivery is bot-based (``MIDAS_DISCORD_BOT_TOKEN`` + a channel id). The
bot replaced an earlier webhook-based delivery entirely: a bot can post the same embeds
AND read reactions, which the confirm flow needs; keeping both meant
two credentials and two send paths for nothing.

The ``OrderConfirmer`` protocol exists so a second transport could back
the pending-order flow later without touching the engine.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal, Protocol

from midas.models import DEFAULT_ALERT_TIMEOUT_SECONDS, AlertsConfig, Direction, Order

logger = logging.getLogger(__name__)

DISCORD_BOT_TOKEN_ENV_VAR = "MIDAS_DISCORD_BOT_TOKEN"
DISCORD_API_BASE = "https://discord.com/api/v10"
# On HTTP 429, honor retry_after once if it's short; otherwise drop the
# request — at a few alerts/day Midas never realistically hits the limit.
MAX_RETRY_AFTER_SECONDS = 5.0
COLOR_BUY = 0x57F287  # Discord green
COLOR_SELL = 0xED4245  # Discord red
# Cloudflare fronts Discord and rejects urllib's default Python-urllib/x.y
# User-Agent with HTTP 403 (error code 1010), so send an explicit one.
USER_AGENT = "midas-alerts/0.2"
# Accepted reaction emoji, compared with the U+FE0F variation selector
# stripped: without pre-seeded reactions the operator picks from the full
# emoji picker, and ✔️ (U+2714) instead of ✅ (U+2705) must not leave a
# confirmed-and-executed trade to silently expire.
CONFIRM_EMOJIS = frozenset({"✅", "✔", "☑"})
DECLINE_EMOJIS = frozenset({"❌", "❎", "✖", "⛔", "🚫"})

# "unavailable" means the poll itself failed (outage, revoked token) —
# distinct from None ("polled fine, no reaction yet") so the engine never
# expires an order it could not actually check.
type Decision = Literal["confirmed", "declined", "unavailable"] | None


class OrderConfirmer(Protocol):
    """Transport for the pending-order confirmation flow.

    Every method must swallow delivery/polling failures and degrade to a
    "nothing happened" result — the live tick never breaks on transport
    errors, and a pending order just waits for the next poll.
    """

    def post_order(self, order: Order, timestamp: datetime) -> str | None:
        """Post one order alert awaiting confirmation.

        Returns:
            An opaque message id to poll later, or None on failure (the
            caller must NOT create a pending order in that case — the
            order will be re-proposed and re-posted on a later tick).
        """

    def poll_decision(self, message_id: str) -> Decision:
        """Return the operator's decision on a posted order, if any yet."""

    def mark_booked(self, message_id: str, note: str) -> None:
        """Annotate a confirmed order's message as booked."""

    def mark_declined(self, message_id: str) -> None:
        """Annotate a declined order's message."""

    def mark_expired(self, message_id: str) -> None:
        """Annotate an expired (unconfirmed past TTL) order's message."""

    def mark_superseded(self, message_id: str) -> None:
        """Annotate an order replaced by a recomputed alert."""

    def mark_unfillable(self, message_id: str, note: str) -> None:
        """Annotate a confirmed order that could not be booked."""


def order_embed(order: Order, timestamp: datetime, *, dry_run: bool = False) -> dict[str, Any]:
    """Build one Discord embed for a sized order."""
    prefix = "[DRY RUN] " if dry_run else ""
    title = f"{prefix}{order.direction.value} {order.ticker} — {order.shares:g} sh @ ${order.price:,.2f}"
    ctx = order.context
    return {
        "title": title,
        "color": COLOR_BUY if order.direction == Direction.BUY else COLOR_SELL,
        "fields": [
            {"name": "Strategy", "value": ctx.source, "inline": True},
            {"name": "Estimated value", "value": f"${order.estimated_value:,.2f}", "inline": True},
            {"name": "Reason", "value": ctx.reason, "inline": False},
        ],
        "timestamp": timestamp.isoformat(),
    }


class DiscordApiError(Exception):
    """A Discord REST call failed — network, HTTP, or malformed response.

    Every failure mode of ``DiscordBotClient`` funnels into this one type
    so callers cannot miss an exotic case (a 200 with a non-JSON
    Cloudflare interstitial body raises ``json.JSONDecodeError``, which is
    NOT an ``OSError`` — catching only ``OSError`` killed the live
    process).

    Attributes:
        status: HTTP status code when the failure was an HTTP error,
            else None.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class DiscordBotClient:
    """Thin REST client for the handful of Discord endpoints Midas uses.

    Every failure raises ``DiscordApiError``; the confirmer above this
    layer owns the never-break-the-tick guarantee. A single short 429
    retry honoring ``Retry-After`` is handled here.
    """

    def __init__(self, bot_token: str, timeout_seconds: float = DEFAULT_ALERT_TIMEOUT_SECONDS) -> None:
        """Store credentials.

        Args:
            bot_token: Discord bot token (a secret — never log it).
            timeout_seconds: Per-request socket timeout.
        """
        self._bot_token = bot_token
        self._timeout_seconds = timeout_seconds

    def create_message(self, channel_id: str, payload: dict[str, Any]) -> str:
        """POST a message to *channel_id*; returns the new message's id."""
        response = self._request("POST", f"/channels/{channel_id}/messages", payload)
        if not isinstance(response, dict) or "id" not in response:
            msg = "Discord message create returned no message id"
            raise DiscordApiError(msg)
        return str(response["id"])

    def edit_message(self, channel_id: str, message_id: str, payload: dict[str, Any]) -> None:
        """PATCH fields (e.g. ``content``) of an existing message."""
        self._request("PATCH", f"/channels/{channel_id}/messages/{message_id}", payload)

    def get_message(self, channel_id: str, message_id: str) -> dict[str, Any]:
        """GET one message (includes its ``reactions`` summary)."""
        response = self._request("GET", f"/channels/{channel_id}/messages/{message_id}")
        if not isinstance(response, dict):
            msg = "Discord message fetch returned a non-object body"
            raise DiscordApiError(msg)
        return response

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        try:
            try:
                return self._request_once(method, path, payload)
            except urllib.error.HTTPError as exc:
                if exc.code != 429:
                    raise
                retry_after = _parse_retry_after(exc)
                if retry_after is None or retry_after > MAX_RETRY_AFTER_SECONDS:
                    raise
                time.sleep(retry_after)
                return self._request_once(method, path, payload)
        except urllib.error.HTTPError as exc:
            raise DiscordApiError(f"HTTP {exc.code}", status=exc.code) from exc
        except (OSError, json.JSONDecodeError) as exc:
            # URLError, timeouts, socket errors — and a 200 whose body is
            # not JSON (Cloudflare interstitials).
            raise DiscordApiError(str(exc)) from exc

    def _request_once(self, method: str, path: str, payload: dict[str, Any] | None) -> Any:
        headers = {
            "Authorization": f"Bot {self._bot_token}",
            "User-Agent": USER_AGENT,
        }
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(f"{DISCORD_API_BASE}{path}", data=data, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
            body = response.read()
        if not body:
            return None
        return json.loads(body)


class DiscordConfirmer:
    """Pending-order confirmation over a Discord channel.

    Posts one embed per order and reads the operator's ✅/❌ reaction
    back. Messages are deliberately bare — no instruction text, no
    pre-seeded reactions (operator preference: the channel stays clean;
    the ✅/❌ convention lives in the docs). All failures are logged and
    swallowed; ``post_order`` degrades to ``None`` and ``poll_decision``
    to "no decision yet".
    """

    def __init__(self, client: DiscordBotClient, channel_id: str) -> None:
        self._client = client
        self._channel_id = channel_id

    def post_order(self, order: Order, timestamp: datetime) -> str | None:
        """Post the order embed awaiting a ✅/❌ reaction."""
        try:
            return self._client.create_message(self._channel_id, {"embeds": [order_embed(order, timestamp)]})
        except DiscordApiError as exc:
            logger.error("Discord alert post failed (%s); will retry on a later tick", exc)
            return None

    def poll_decision(self, message_id: str) -> Decision:
        """Check for an operator ✅ or ❌ on the message.

        One message fetch reads the full reaction summary. Check-mark
        variants count as confirm and cross variants as decline (the
        operator picks from the raw emoji picker — see CONFIRM_EMOJIS);
        anything else is logged so a near-miss reaction is never silent.
        If both are present, confirm wins: an executed trade must be
        booked — un-booking a real fill corrupts state, while an
        accidental extra ❌ costs nothing.

        Returns:
            "confirmed"/"declined" on an operator reaction; None when the
            message was polled fine but carries no decision (or was
            deleted — HTTP 404 — so the normal TTL expiry applies);
            "unavailable" when the poll itself failed, which the engine
            must never treat as "no reaction".
        """
        try:
            message = self._client.get_message(self._channel_id, message_id)
        except DiscordApiError as exc:
            if exc.status == 404:
                # Message deleted — it can never be confirmed; let the
                # TTL expire it normally.
                return None
            logger.error("Discord reaction poll failed (%s); will retry next tick", exc)
            return "unavailable"
        confirmed = False
        declined = False
        for reaction in message.get("reactions") or []:
            emoji = reaction.get("emoji") or {}
            name = str(emoji.get("name") or "").replace("\ufe0f", "")
            if int(reaction.get("count") or 0) <= 0:
                continue
            if name in CONFIRM_EMOJIS:
                confirmed = True
            elif name in DECLINE_EMOJIS:
                declined = True
            else:
                logger.warning("unrecognized reaction %r on alert message — react \u2705 or \u274c", name)
        if confirmed:
            return "confirmed"
        if declined:
            return "declined"
        return None

    def mark_booked(self, message_id: str, note: str) -> None:
        self._edit_content(message_id, f"\u2705 Booked — {note}")

    def mark_declined(self, message_id: str) -> None:
        self._edit_content(message_id, "\u274c Declined — not executed")

    def mark_expired(self, message_id: str) -> None:
        self._edit_content(message_id, "\u23f0 Expired — no longer confirmable")

    def mark_superseded(self, message_id: str) -> None:
        self._edit_content(message_id, "\U0001f504 Superseded — replaced by a newer alert")

    def mark_unfillable(self, message_id: str, note: str) -> None:
        self._edit_content(message_id, f"\u26a0\ufe0f Confirmed but NOT booked — {note}")

    def _edit_content(self, message_id: str, content: str) -> None:
        try:
            self._client.edit_message(self._channel_id, message_id, {"content": content})
        except DiscordApiError as exc:
            # Cosmetic — the state transition already happened.
            logger.warning("failed to annotate alert message (%s)", exc)


def _parse_retry_after(exc: urllib.error.HTTPError) -> float | None:
    """Extract the retry delay in seconds from a Discord 429 response."""
    header = exc.headers.get("Retry-After") if exc.headers is not None else None
    if header is None:
        return None
    try:
        return float(header)
    except ValueError:
        return None


COLOR_REPORT = 0x5865F2  # Discord blurple — neither buy-green nor sell-red


@dataclass(frozen=True)
class DailyReport:
    """Close-of-day summary assembled by the live engine."""

    session_date: date
    equity: float
    previous_equity: float | None
    cash: float
    positions: int
    pending: int
    alerts_posted: int
    confirmed: int
    declined: int
    expired: int
    cppi_scale: float
    vol_target_scale: float


def report_embed(report: DailyReport) -> dict[str, Any]:
    """Build the close-of-day report embed (plain: no reactions, never polled)."""
    if report.previous_equity is not None and report.previous_equity > 0:
        delta = report.equity - report.previous_equity
        pct = delta / report.previous_equity
        day_change = f"{'+' if delta >= 0 else '-'}${abs(delta):,.2f} ({pct:+.2%})"
    else:
        day_change = "n/a (first report)"
    alerts_line = (
        f"{report.alerts_posted} posted · {report.confirmed} \u2705 · "
        f"{report.declined} \u274c · {report.expired} expired · {report.pending} pending"
    )
    fields = [
        {"name": "Equity", "value": f"${report.equity:,.2f}", "inline": True},
        {"name": "Day", "value": day_change, "inline": True},
        {"name": "Cash", "value": f"${report.cash:,.2f}", "inline": True},
        {"name": "Positions", "value": str(report.positions), "inline": True},
        {"name": "Alerts", "value": alerts_line, "inline": False},
    ]
    if report.cppi_scale != 1.0 or report.vol_target_scale != 1.0:
        fields.append(
            {
                "name": "Risk overlays",
                "value": f"CPPI \u00d7{report.cppi_scale:.2f} · vol target \u00d7{report.vol_target_scale:.2f}",
                "inline": False,
            }
        )
    return {
        "title": f"Close of day — {report.session_date.isoformat()}",
        "color": COLOR_REPORT,
        "fields": fields,
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }


class DiscordReporter:
    """Posts informational embeds (close-of-day report) to a channel.

    Plain messages: no reactions, never polled, and — as everywhere in
    this module — a delivery failure is logged and swallowed, never
    raised into the tick loop.
    """

    def __init__(self, client: DiscordBotClient, channel_id: str) -> None:
        self._client = client
        self._channel_id = channel_id

    def post_report(self, report: DailyReport) -> bool:
        """Post the report embed. Returns False on delivery failure."""
        try:
            self._client.create_message(self._channel_id, {"embeds": [report_embed(report)]})
        except DiscordApiError as exc:
            logger.error("Discord report delivery failed (%s)", exc)
            return False
        return True


def build_reporter(alerts_config: AlertsConfig | None) -> DiscordReporter | None:
    """Resolve the close-of-day reporter, or None for terminal-only.

    Rides the same bot credentials as the confirmer. Reports go to
    ``discord_report_channel_id`` when set, else fall back to the alerts
    channel. No bot configured -> terminal-only reporting.
    """
    token = os.environ.get(DISCORD_BOT_TOKEN_ENV_VAR, "").strip()
    if not token or alerts_config is None:
        return None
    channel_id = (alerts_config.discord_report_channel_id or alerts_config.discord_channel_id).strip()
    if not channel_id:
        return None
    return DiscordReporter(
        DiscordBotClient(token, timeout_seconds=alerts_config.timeout_seconds),
        channel_id,
    )


def build_confirmer(alerts_config: AlertsConfig | None) -> DiscordConfirmer | None:
    """Resolve the Discord confirmer for a live run, or None for terminal-only.

    Discord delivery requires BOTH the ``MIDAS_DISCORD_BOT_TOKEN`` env var
    (the token is a secret — env only, never YAML) and ``discord_channel_id``
    in the ``alerts:`` block. Neither → terminal-only with assumed fills,
    behavior unchanged from before Discord existed.

    Raises:
        ValueError: If exactly one of the two is configured — a half-wired
            bot must be a loud startup error, not a silent fallback.
    """
    token = os.environ.get(DISCORD_BOT_TOKEN_ENV_VAR, "").strip()
    channel_id = alerts_config.discord_channel_id.strip() if alerts_config is not None else ""
    if not token and not channel_id:
        return None
    if not token:
        msg = (
            f"alerts: discord_channel_id is set but {DISCORD_BOT_TOKEN_ENV_VAR} is not — "
            "export the bot token or remove the channel id"
        )
        raise ValueError(msg)
    if not channel_id:
        msg = (
            f"{DISCORD_BOT_TOKEN_ENV_VAR} is set but alerts: discord_channel_id is not — "
            "add the channel id to the portfolio YAML or unset the token"
        )
        raise ValueError(msg)
    timeout = alerts_config.timeout_seconds if alerts_config is not None else DEFAULT_ALERT_TIMEOUT_SECONDS
    return DiscordConfirmer(DiscordBotClient(token, timeout_seconds=timeout), channel_id)
