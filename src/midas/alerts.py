"""Discord delivery and fill confirmation for live alerts (issues #73/#81).

The terminal remains the alert channel of record — Discord is additive.
A delivery or polling failure must never break a live tick: every network
error is logged and swallowed at this module's public surface, mirroring
the trade-log salvage philosophy in ``live.py``.

Delivery is bot-based (``MIDAS_DISCORD_BOT_TOKEN`` + a channel id). The
bot replaced the #73 webhook entirely: a bot can post the same embeds
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
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Literal, Protocol

from midas.models import AlertsConfig, Direction, Order

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
CONFIRM_EMOJI = "✅"
DECLINE_EMOJI = "❌"

type Decision = Literal["confirmed", "declined"] | None


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
        """Annotate an expired/superseded order's message."""


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


class DiscordBotClient:
    """Thin REST client for the handful of Discord endpoints Midas uses.

    Methods raise ``OSError`` subclasses (``HTTPError``/``URLError``/
    timeouts) on failure; the confirmer above this layer owns the
    never-break-the-tick guarantee. A single short 429 retry honoring
    ``Retry-After`` is handled here.
    """

    def __init__(self, bot_token: str, timeout_seconds: float = 5.0) -> None:
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
        return str(response["id"])

    def edit_message(self, channel_id: str, message_id: str, payload: dict[str, Any]) -> None:
        """PATCH fields (e.g. ``content``) of an existing message."""
        self._request("PATCH", f"/channels/{channel_id}/messages/{message_id}", payload)

    def get_reaction_users(self, channel_id: str, message_id: str, emoji: str) -> list[dict[str, Any]]:
        """List the user objects that reacted with *emoji* on a message."""
        encoded = urllib.parse.quote(emoji)
        response = self._request("GET", f"/channels/{channel_id}/messages/{message_id}/reactions/{encoded}")
        return response if isinstance(response, list) else []

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
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
        except OSError as exc:
            logger.error("Discord alert post failed (%s); will retry on a later tick", _describe(exc))
            return None

    def poll_decision(self, message_id: str) -> Decision:
        """Check for a human ✅ or ❌ on the message.

        Reactions from bots are ignored via the ``bot`` flag on Discord
        user objects — only a human's tap counts. If both are present,
        ✅ wins: an executed trade must be booked — un-booking a real
        fill corrupts state, while an accidental extra ❌ costs nothing.
        """
        if self._human_reacted(message_id, CONFIRM_EMOJI):
            return "confirmed"
        if self._human_reacted(message_id, DECLINE_EMOJI):
            return "declined"
        return None

    def mark_booked(self, message_id: str, note: str) -> None:
        self._edit_content(message_id, f"✅ Booked — {note}")

    def mark_declined(self, message_id: str) -> None:
        self._edit_content(message_id, "❌ Declined — not executed")

    def mark_expired(self, message_id: str) -> None:
        self._edit_content(message_id, "⏰ Expired — no longer confirmable")

    def _human_reacted(self, message_id: str, emoji: str) -> bool:
        try:
            users = self._client.get_reaction_users(self._channel_id, message_id, emoji)
        except OSError as exc:
            logger.error("Discord reaction poll failed (%s); will retry next tick", _describe(exc))
            return False
        return any(not user.get("bot", False) for user in users)

    def _edit_content(self, message_id: str, content: str) -> None:
        try:
            self._client.edit_message(self._channel_id, message_id, {"content": content})
        except OSError as exc:
            # Cosmetic — the state transition already happened.
            logger.warning("failed to annotate alert message (%s)", _describe(exc))


def _describe(exc: OSError) -> str:
    """Compact error description; HTTP status for HTTPError, str otherwise."""
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    return str(exc)


def _parse_retry_after(exc: urllib.error.HTTPError) -> float | None:
    """Extract the retry delay in seconds from a Discord 429 response."""
    header = exc.headers.get("Retry-After") if exc.headers is not None else None
    if header is None:
        return None
    try:
        return float(header)
    except ValueError:
        return None


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
    timeout = alerts_config.timeout_seconds if alerts_config is not None else 5.0
    return DiscordConfirmer(DiscordBotClient(token, timeout_seconds=timeout), channel_id)
