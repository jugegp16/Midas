"""Tests for Discord bot delivery and confirmation (midas.alerts)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import UTC, datetime
from email.message import Message
from typing import Any

import pytest

from midas import alerts
from midas.alerts import (
    DISCORD_BOT_TOKEN_ENV_VAR,
    DiscordApiError,
    DiscordBotClient,
    DiscordConfirmer,
    build_confirmer,
    order_embed,
)
from midas.models import AlertsConfig, Direction, Order, OrderContext

CHANNEL = "1400000000000000001"
NOW = datetime(2026, 8, 19, 14, 30, 0, tzinfo=UTC)


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


def http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://discord.com/api/v10/x", code, "err", headers, None)


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture every Discord REST request; respond with a message object."""
    requests: list[dict[str, Any]] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> FakeResponse:
        requests.append(
            {
                "method": request.get_method(),
                "url": request.full_url,
                "payload": json.loads(request.data) if request.data else None,
                "headers": dict(request.header_items()),
                "timeout": timeout,
            }
        )
        return FakeResponse(b'{"id": "999888777"}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return requests


class TestEmbedFormat:
    def test_buy_embed(self) -> None:
        embed = order_embed(make_order(), NOW)

        assert embed["title"] == "BUY AAPL — 10.5 sh @ $30.12"
        assert embed["color"] == alerts.COLOR_BUY
        assert embed["timestamp"] == NOW.isoformat()
        fields = {field["name"]: field["value"] for field in embed["fields"]}
        assert fields["Strategy"] == "Momentum"
        assert fields["Reason"] == "momentum crossover"
        assert fields["Estimated value"] == "$316.26"

    def test_whole_share_orders_print_as_integers(self) -> None:
        """The sizer floors to whole shares — no '23.0000 sh' noise."""
        embed = order_embed(make_order(shares=23.0, price=100.0), NOW)

        assert embed["title"] == "BUY AAPL — 23 sh @ $100.00"

    def test_sell_embed_is_red(self) -> None:
        embed = order_embed(make_order(direction=Direction.SELL), NOW)

        assert embed["title"].startswith("SELL AAPL")
        assert embed["color"] == alerts.COLOR_SELL

    def test_dry_run_prefix(self) -> None:
        embed = order_embed(make_order(), NOW, dry_run=True)

        assert embed["title"].startswith("[DRY RUN] BUY AAPL")


class TestBotClient:
    def test_create_message_request_shape(self, captured: list[dict[str, Any]]) -> None:
        client = DiscordBotClient("tok-123", timeout_seconds=2.5)

        message_id = client.create_message(CHANNEL, {"content": "hi"})

        assert message_id == "999888777"
        req = captured[0]
        assert req["method"] == "POST"
        assert req["url"] == f"{alerts.DISCORD_API_BASE}/channels/{CHANNEL}/messages"
        assert req["payload"] == {"content": "hi"}
        assert req["headers"]["Authorization"] == "Bot tok-123"
        assert req["headers"]["User-agent"] == alerts.USER_AGENT
        assert req["timeout"] == 2.5

    def test_edit_message_patches_content(self, captured: list[dict[str, Any]]) -> None:
        DiscordBotClient("tok").edit_message(CHANNEL, "42", {"content": "done"})

        req = captured[0]
        assert req["method"] == "PATCH"
        assert req["url"].endswith(f"/channels/{CHANNEL}/messages/42")
        assert req["payload"] == {"content": "done"}

    def test_get_message_returns_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        body = json.dumps({"id": "42", "reactions": []}).encode()
        monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=None: FakeResponse(body))

        message = DiscordBotClient("tok").get_message(CHANNEL, "42")

        assert message["id"] == "42"

    def test_non_json_200_body_raises_discord_api_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A Cloudflare interstitial (200 + HTML) must not escape as JSONDecodeError."""
        monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=None: FakeResponse(b"<html>nope</html>"))

        with pytest.raises(DiscordApiError):
            DiscordBotClient("tok").get_message(CHANNEL, "42")

    def test_missing_message_id_raises_discord_api_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=None: FakeResponse(b'{"oops": 1}'))

        with pytest.raises(DiscordApiError):
            DiscordBotClient("tok").create_message(CHANNEL, {"content": "hi"})

    def test_network_error_raises_discord_api_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> FakeResponse:
            raise urllib.error.URLError("down")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        with pytest.raises(DiscordApiError):
            DiscordBotClient("tok").get_message(CHANNEL, "42")

    def test_http_error_carries_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> FakeResponse:
            raise http_error(404)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        with pytest.raises(DiscordApiError) as excinfo:
            DiscordBotClient("tok").get_message(CHANNEL, "42")
        assert excinfo.value.status == 404

    def test_429_with_short_retry_after_retries_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []
        sleeps: list[float] = []

        def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> FakeResponse:
            calls.append("req")
            if len(calls) == 1:
                raise http_error(429, "0.7")
            return FakeResponse(b'{"id": "1"}')

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(alerts.time, "sleep", sleeps.append)

        DiscordBotClient("tok").create_message(CHANNEL, {"content": "hi"})

        assert calls == ["req", "req"]
        assert sleeps == [0.7]

    def test_429_with_long_retry_after_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> FakeResponse:
            raise http_error(429, "30")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        with pytest.raises(DiscordApiError) as excinfo:
            DiscordBotClient("tok").create_message(CHANNEL, {"content": "hi"})
        assert excinfo.value.status == 429


class TestConfirmer:
    def make(self, monkeypatch: pytest.MonkeyPatch, responder: Any) -> DiscordConfirmer:
        monkeypatch.setattr(urllib.request, "urlopen", responder)
        return DiscordConfirmer(DiscordBotClient("tok"), CHANNEL)

    def _message_responder(self, reactions: list[dict[str, Any]]) -> Any:
        body = json.dumps({"id": "42", "reactions": reactions}).encode()
        return lambda request, timeout=None: FakeResponse(body)

    def test_post_order_posts_bare_embed_only(self, captured: list[dict[str, Any]]) -> None:
        """One POST, embed only: no instruction content, no seeded reactions."""
        confirmer = DiscordConfirmer(DiscordBotClient("tok"), CHANNEL)

        message_id = confirmer.post_order(make_order(), NOW)

        assert message_id == "999888777"
        assert [req["method"] for req in captured] == ["POST"]
        assert captured[0]["payload"]["embeds"][0]["title"].startswith("BUY AAPL")
        assert "content" not in captured[0]["payload"]

    def test_post_order_failure_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> FakeResponse:
            raise urllib.error.URLError("down")

        confirmer = self.make(monkeypatch, fake_urlopen)

        assert confirmer.post_order(make_order(), NOW) is None

    def test_post_order_non_json_response_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The failure that killed the live process: 200 + non-JSON body."""
        confirmer = self.make(monkeypatch, lambda request, timeout=None: FakeResponse(b"<html>cf</html>"))

        assert confirmer.post_order(make_order(), NOW) is None  # must not raise

    def test_poll_no_reactions_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        confirmer = self.make(monkeypatch, self._message_responder([]))

        assert confirmer.poll_decision("42") is None

    def test_poll_confirm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        confirmer = self.make(monkeypatch, self._message_responder([{"emoji": {"name": "\u2705"}, "count": 1}]))

        assert confirmer.poll_decision("42") == "confirmed"

    def test_poll_accepts_check_mark_variants(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """\u2714\ufe0f from the raw picker must count — a confirmed-and-executed
        trade must not silently expire over a near-miss codepoint."""
        confirmer = self.make(monkeypatch, self._message_responder([{"emoji": {"name": "\u2714\ufe0f"}, "count": 1}]))

        assert confirmer.poll_decision("42") == "confirmed"

    def test_poll_decline_and_cross_variants(self, monkeypatch: pytest.MonkeyPatch) -> None:
        confirmer = self.make(monkeypatch, self._message_responder([{"emoji": {"name": "\u274c"}, "count": 1}]))
        assert confirmer.poll_decision("42") == "declined"

        confirmer = self.make(monkeypatch, self._message_responder([{"emoji": {"name": "\u2716\ufe0f"}, "count": 1}]))
        assert confirmer.poll_decision("42") == "declined"

    def test_poll_unrecognized_reaction_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        confirmer = self.make(monkeypatch, self._message_responder([{"emoji": {"name": "\U0001f44d"}, "count": 1}]))

        assert confirmer.poll_decision("42") is None

    def test_poll_confirm_wins_over_decline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both tapped: an executed trade must be booked."""
        confirmer = self.make(
            monkeypatch,
            self._message_responder(
                [{"emoji": {"name": "\u274c"}, "count": 1}, {"emoji": {"name": "\u2705"}, "count": 1}]
            ),
        )

        assert confirmer.poll_decision("42") == "confirmed"

    def test_poll_failure_is_unavailable_not_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An outage must be distinguishable from "no reaction yet" — the
        engine must never TTL-expire an order it could not check."""

        def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> FakeResponse:
            raise http_error(500)

        confirmer = self.make(monkeypatch, fake_urlopen)

        assert confirmer.poll_decision("42") == "unavailable"

    def test_poll_deleted_message_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """404 = message gone, never confirmable — normal TTL expiry applies."""

        def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> FakeResponse:
            raise http_error(404)

        confirmer = self.make(monkeypatch, fake_urlopen)

        assert confirmer.poll_decision("42") is None

    def test_marks_edit_content_and_swallow_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> FakeResponse:
            raise http_error(404)

        confirmer = self.make(monkeypatch, fake_urlopen)

        confirmer.mark_booked("42", "note")  # must not raise
        confirmer.mark_declined("42")
        confirmer.mark_expired("42")
        confirmer.mark_superseded("42")
        confirmer.mark_unfillable("42", "note")


class TestBuildConfirmer:
    def test_neither_configured_is_terminal_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(DISCORD_BOT_TOKEN_ENV_VAR, raising=False)

        assert build_confirmer(None) is None
        assert build_confirmer(AlertsConfig()) is None

    def test_channel_without_token_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(DISCORD_BOT_TOKEN_ENV_VAR, raising=False)

        with pytest.raises(ValueError, match=DISCORD_BOT_TOKEN_ENV_VAR):
            build_confirmer(AlertsConfig(discord_channel_id=CHANNEL))

    def test_token_without_channel_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DISCORD_BOT_TOKEN_ENV_VAR, "tok")

        with pytest.raises(ValueError, match="discord_channel_id"):
            build_confirmer(AlertsConfig())
        with pytest.raises(ValueError, match="discord_channel_id"):
            build_confirmer(None)

    def test_both_configured_builds_confirmer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DISCORD_BOT_TOKEN_ENV_VAR, "tok")

        confirmer = build_confirmer(AlertsConfig(discord_channel_id=CHANNEL, timeout_seconds=3.0))

        assert isinstance(confirmer, DiscordConfirmer)
        assert confirmer._channel_id == CHANNEL
        assert confirmer._client._timeout_seconds == 3.0

    def test_blank_token_is_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DISCORD_BOT_TOKEN_ENV_VAR, "   ")

        assert build_confirmer(AlertsConfig()) is None
