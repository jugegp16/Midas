"""Tests for Discord alert delivery (midas.alerts)."""

from __future__ import annotations

import urllib.error
import urllib.request
from datetime import UTC, datetime
from email.message import Message
from typing import Any
from unittest.mock import MagicMock

import pytest

from midas import alerts
from midas.alerts import (
    DISCORD_WEBHOOK_ENV_VAR,
    MAX_EMBEDS_PER_POST,
    DiscordAlertSink,
    build_alert_sinks,
)
from midas.models import AlertsConfig, Direction, Order, OrderContext

WEBHOOK = "https://discord.com/api/webhooks/123/abc"
NOW = datetime(2026, 8, 18, 14, 30, 0, tzinfo=UTC)


def make_order(
    ticker: str = "AAPL",
    direction: Direction = Direction.BUY,
    shares: float = 10.5,
    price: float = 30.12,
) -> Order:
    return Order(
        ticker=ticker,
        direction=direction,
        shares=shares,
        price=price,
        estimated_value=shares * price,
        context=OrderContext(
            contributions={},
            blended_score=0.5,
            target_weight=0.2,
            current_weight=0.1,
            reason="momentum crossover",
            source="Momentum",
        ),
    )


def http_429(retry_after: str | None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(WEBHOOK, 429, "Too Many Requests", headers, None)


@pytest.fixture
def captured_posts(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture every webhook POST payload instead of hitting the network."""
    import json

    posts: list[dict[str, Any]] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> MagicMock:
        posts.append(
            {
                "payload": json.loads(request.data),
                "timeout": timeout,
                "url": request.full_url,
                "headers": dict(request.header_items()),
            }
        )
        return MagicMock()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return posts


class TestEmbedFormat:
    def test_buy_embed(self, captured_posts: list[dict[str, Any]]) -> None:
        DiscordAlertSink(WEBHOOK).send_orders([make_order()], NOW)

        assert len(captured_posts) == 1
        embed = captured_posts[0]["payload"]["embeds"][0]
        assert embed["title"] == "BUY AAPL — 10.5000 sh @ $30.12"
        assert embed["color"] == alerts.COLOR_BUY
        assert embed["timestamp"] == NOW.isoformat()
        fields = {field["name"]: field["value"] for field in embed["fields"]}
        assert fields["Strategy"] == "Momentum"
        assert fields["Reason"] == "momentum crossover"
        assert fields["Estimated value"] == "$316.26"

    def test_sell_embed_is_red(self, captured_posts: list[dict[str, Any]]) -> None:
        DiscordAlertSink(WEBHOOK).send_orders([make_order(direction=Direction.SELL)], NOW)

        embed = captured_posts[0]["payload"]["embeds"][0]
        assert embed["title"].startswith("SELL AAPL")
        assert embed["color"] == alerts.COLOR_SELL

    def test_dry_run_prefix(self, captured_posts: list[dict[str, Any]]) -> None:
        DiscordAlertSink(WEBHOOK).send_orders([make_order()], NOW, dry_run=True)

        embed = captured_posts[0]["payload"]["embeds"][0]
        assert embed["title"].startswith("[DRY RUN] BUY AAPL")

    def test_timeout_and_url_are_used(self, captured_posts: list[dict[str, Any]]) -> None:
        DiscordAlertSink(WEBHOOK, timeout_seconds=2.5).send_orders([make_order()], NOW)

        assert captured_posts[0]["url"] == WEBHOOK
        assert captured_posts[0]["timeout"] == 2.5

    def test_explicit_user_agent_is_sent(self, captured_posts: list[dict[str, Any]]) -> None:
        """Cloudflare 403s urllib's default UA (error 1010) — pin the override."""
        DiscordAlertSink(WEBHOOK).send_orders([make_order()], NOW)

        headers = captured_posts[0]["headers"]
        assert headers.get("User-agent") == alerts.USER_AGENT


class TestBatching:
    def test_orders_batch_at_discord_cap(self, captured_posts: list[dict[str, Any]]) -> None:
        orders = [make_order(ticker=f"T{i:02d}") for i in range(25)]

        DiscordAlertSink(WEBHOOK).send_orders(orders, NOW)

        sizes = [len(post["payload"]["embeds"]) for post in captured_posts]
        assert sizes == [MAX_EMBEDS_PER_POST, MAX_EMBEDS_PER_POST, 5]

    def test_no_orders_no_post(self, captured_posts: list[dict[str, Any]]) -> None:
        DiscordAlertSink(WEBHOOK).send_orders([], NOW)

        assert captured_posts == []


class TestFailureSemantics:
    def test_network_error_is_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> MagicMock:
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        DiscordAlertSink(WEBHOOK).send_orders([make_order()], NOW)  # must not raise

    def test_non_429_http_error_is_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> MagicMock:
            raise urllib.error.HTTPError(WEBHOOK, 404, "Not Found", Message(), None)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        DiscordAlertSink(WEBHOOK).send_orders([make_order()], NOW)  # must not raise

    def test_429_with_short_retry_after_retries_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []
        sleeps: list[float] = []

        def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> MagicMock:
            calls.append("post")
            if len(calls) == 1:
                raise http_429("0.7")
            return MagicMock()

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(alerts.time, "sleep", sleeps.append)

        DiscordAlertSink(WEBHOOK).send_orders([make_order()], NOW)

        assert calls == ["post", "post"]
        assert sleeps == [0.7]

    def test_429_with_long_retry_after_drops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []

        def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> MagicMock:
            calls.append("post")
            raise http_429("30")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        DiscordAlertSink(WEBHOOK).send_orders([make_order()], NOW)

        assert calls == ["post"]

    def test_429_without_retry_after_drops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []

        def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> MagicMock:
            calls.append("post")
            raise http_429(None)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        DiscordAlertSink(WEBHOOK).send_orders([make_order()], NOW)

        assert calls == ["post"]

    def test_429_retry_failure_is_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> MagicMock:
            raise http_429("0.1")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(alerts.time, "sleep", lambda _s: None)

        DiscordAlertSink(WEBHOOK).send_orders([make_order()], NOW)  # must not raise


class TestBuildAlertSinks:
    def test_no_config_no_env_no_sinks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(DISCORD_WEBHOOK_ENV_VAR, raising=False)

        assert build_alert_sinks(None) == []

    def test_config_without_url_no_sinks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(DISCORD_WEBHOOK_ENV_VAR, raising=False)

        assert build_alert_sinks(AlertsConfig()) == []

    def test_yaml_url_builds_discord_sink(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(DISCORD_WEBHOOK_ENV_VAR, raising=False)

        sinks = build_alert_sinks(AlertsConfig(discord_webhook_url=WEBHOOK, timeout_seconds=3.0))

        assert len(sinks) == 1
        sink = sinks[0]
        assert isinstance(sink, DiscordAlertSink)
        assert sink._webhook_url == WEBHOOK
        assert sink._timeout_seconds == 3.0

    def test_env_var_overrides_yaml_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        env_url = "https://discord.com/api/webhooks/999/env"
        monkeypatch.setenv(DISCORD_WEBHOOK_ENV_VAR, env_url)

        sinks = build_alert_sinks(AlertsConfig(discord_webhook_url=WEBHOOK))

        assert isinstance(sinks[0], DiscordAlertSink)
        assert sinks[0]._webhook_url == env_url

    def test_env_var_alone_enables_sink_with_default_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DISCORD_WEBHOOK_ENV_VAR, WEBHOOK)

        sinks = build_alert_sinks(None)

        assert len(sinks) == 1
        assert isinstance(sinks[0], DiscordAlertSink)
        assert sinks[0]._timeout_seconds == 5.0

    def test_blank_env_var_falls_back_to_yaml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DISCORD_WEBHOOK_ENV_VAR, "   ")

        sinks = build_alert_sinks(AlertsConfig(discord_webhook_url=WEBHOOK))

        assert isinstance(sinks[0], DiscordAlertSink)
        assert sinks[0]._webhook_url == WEBHOOK

    def test_schemeless_yaml_url_raises_without_leaking_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A scheme-less URL must fail at startup, and the secret token must not appear in the error."""
        monkeypatch.delenv(DISCORD_WEBHOOK_ENV_VAR, raising=False)

        with pytest.raises(ValueError, match="https://") as excinfo:
            build_alert_sinks(AlertsConfig(discord_webhook_url="discord.com/api/webhooks/1/secrettoken"))

        assert "secrettoken" not in str(excinfo.value)

    def test_schemeless_env_url_raises_without_leaking_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DISCORD_WEBHOOK_ENV_VAR, "discord.com/api/webhooks/1/secrettoken")

        with pytest.raises(ValueError, match="https://") as excinfo:
            build_alert_sinks(None)

        assert "secrettoken" not in str(excinfo.value)
