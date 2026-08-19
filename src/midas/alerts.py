"""Push-notification alert sinks for the live engine (issue #73).

The terminal remains the alert channel of record — sinks are additive.
A sink failure must never break a live tick: every network error is
logged and swallowed inside the sink, mirroring the trade-log salvage
philosophy in ``live.py``.

Only Discord is implemented. The ``AlertSink`` protocol exists so a
second transport can be added later without touching the engine.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from midas.models import AlertsConfig, Direction, Order

logger = logging.getLogger(__name__)

DISCORD_WEBHOOK_ENV_VAR = "MIDAS_DISCORD_WEBHOOK"
# Discord rejects payloads with more than 10 embeds per POST.
MAX_EMBEDS_PER_POST = 10
# On HTTP 429, honor retry_after once if it's short; otherwise drop the
# batch — at a few alerts/day Midas never realistically hits the limit.
MAX_RETRY_AFTER_SECONDS = 5.0
COLOR_BUY = 0x57F287  # Discord green
COLOR_SELL = 0xED4245  # Discord red


class AlertSink(Protocol):
    """A push-notification transport for live order alerts."""

    def send_orders(self, orders: Sequence[Order], timestamp: datetime, *, dry_run: bool = False) -> None:
        """Deliver alerts for *orders*. Must never raise on delivery failure."""


def _order_embed(order: Order, timestamp: datetime, *, dry_run: bool) -> dict[str, object]:
    """Build one Discord embed for a sized order."""
    prefix = "[DRY RUN] " if dry_run else ""
    title = f"{prefix}{order.direction.value} {order.ticker} — {order.shares:.4f} sh @ ${order.price:,.2f}"
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


class DiscordAlertSink:
    """Delivers order alerts to a Discord channel via an incoming webhook.

    All delivery failures are logged and swallowed — the live tick must
    survive Discord being down, the URL being revoked, or the network
    being absent.
    """

    def __init__(self, webhook_url: str, timeout_seconds: float = 5.0) -> None:
        """Store the webhook target.

        Args:
            webhook_url: Discord incoming-webhook URL (a write-capability
                secret — never log it).
            timeout_seconds: Per-POST socket timeout.
        """
        self._webhook_url = webhook_url
        self._timeout_seconds = timeout_seconds

    def send_orders(self, orders: Sequence[Order], timestamp: datetime, *, dry_run: bool = False) -> None:
        """Post one embed per order, batched to Discord's per-POST cap."""
        embeds = [_order_embed(order, timestamp, dry_run=dry_run) for order in orders]
        for start in range(0, len(embeds), MAX_EMBEDS_PER_POST):
            self._post_embeds(embeds[start : start + MAX_EMBEDS_PER_POST])

    def _post_embeds(self, embeds: list[dict[str, object]]) -> None:
        """POST one batch, honoring a single short 429 retry. Never raises."""
        try:
            self._post_once(embeds)
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                logger.error("Discord alert delivery failed (HTTP %s); alert remains in terminal output", exc.code)
                return
            retry_after = _parse_retry_after(exc)
            if retry_after is None or retry_after > MAX_RETRY_AFTER_SECONDS:
                logger.error("Discord rate limit (retry_after=%s); dropping %d alert(s)", retry_after, len(embeds))
                return
            time.sleep(retry_after)
            try:
                self._post_once(embeds)
            except OSError as retry_exc:
                logger.error("Discord alert delivery failed after 429 retry: %s", retry_exc)
        except OSError as exc:
            # URLError, timeouts, and socket errors are all OSError subclasses.
            logger.error("Discord alert delivery failed (%s); alert remains in terminal output", exc)

    def _post_once(self, embeds: list[dict[str, object]]) -> None:
        payload = json.dumps({"username": "Midas", "embeds": embeds}).encode("utf-8")
        request = urllib.request.Request(
            self._webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout_seconds):
            pass


def _parse_retry_after(exc: urllib.error.HTTPError) -> float | None:
    """Extract the retry delay in seconds from a Discord 429 response.

    Discord sends both a ``Retry-After`` header and a ``retry_after`` JSON
    body field; the header is authoritative and cheaper to read.
    """
    header = exc.headers.get("Retry-After") if exc.headers is not None else None
    if header is None:
        return None
    try:
        return float(header)
    except ValueError:
        return None


def build_alert_sinks(alerts_config: AlertsConfig | None) -> list[AlertSink]:
    """Resolve configured push-notification sinks for a live run.

    The ``MIDAS_DISCORD_WEBHOOK`` env var overrides (or stands in for) the
    YAML ``discord_webhook_url``, so the secret URL never has to live in a
    committed portfolio file. No URL from either source → no sinks, and
    live behavior is byte-identical to before alerts existed.
    """
    webhook_url = os.environ.get(DISCORD_WEBHOOK_ENV_VAR, "").strip()
    if not webhook_url and alerts_config is not None:
        webhook_url = alerts_config.discord_webhook_url.strip()
    if not webhook_url:
        return []
    timeout = alerts_config.timeout_seconds if alerts_config is not None else 5.0
    return [DiscordAlertSink(webhook_url, timeout_seconds=timeout)]
