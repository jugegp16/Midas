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
    CONFIRM_EMOJI,
    DISCORD_BOT_TOKEN_ENV_VAR,
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

    def test_get_reaction_users_parses_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        body = json.dumps([{"id": "1", "bot": True}, {"id": "2"}]).encode()
        monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=None: FakeResponse(body))

        users = DiscordBotClient("tok").get_reaction_users(CHANNEL, "42", CONFIRM_EMOJI)

        assert users == [{"id": "1", "bot": True}, {"id": "2"}]

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

        with pytest.raises(urllib.error.HTTPError):
            DiscordBotClient("tok").create_message(CHANNEL, {"content": "hi"})


class TestConfirmer:
    def make(self, monkeypatch: pytest.MonkeyPatch, responder: Any) -> DiscordConfirmer:
        monkeypatch.setattr(urllib.request, "urlopen", responder)
        return DiscordConfirmer(DiscordBotClient("tok"), CHANNEL)

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

    def _reaction_responder(self, confirm_users: list[dict[str, Any]], decline_users: list[dict[str, Any]]) -> Any:
        def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> FakeResponse:
            if "%E2%9C%85" in request.full_url:
                return FakeResponse(json.dumps(confirm_users).encode())
            return FakeResponse(json.dumps(decline_users).encode())

        return fake_urlopen

    def test_poll_ignores_bot_seed_reactions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bot_only = [{"id": "1", "bot": True}]
        confirmer = self.make(monkeypatch, self._reaction_responder(bot_only, bot_only))

        assert confirmer.poll_decision("42") is None

    def test_poll_human_confirm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        confirmer = self.make(
            monkeypatch,
            self._reaction_responder([{"id": "1", "bot": True}, {"id": "2"}], [{"id": "1", "bot": True}]),
        )

        assert confirmer.poll_decision("42") == "confirmed"

    def test_poll_human_decline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        confirmer = self.make(
            monkeypatch,
            self._reaction_responder([{"id": "1", "bot": True}], [{"id": "1", "bot": True}, {"id": "2"}]),
        )

        assert confirmer.poll_decision("42") == "declined"

    def test_poll_confirm_wins_over_decline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both tapped: an executed trade must be booked."""
        both = [{"id": "2"}]
        confirmer = self.make(monkeypatch, self._reaction_responder(both, both))

        assert confirmer.poll_decision("42") == "confirmed"

    def test_poll_failure_means_no_decision(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> FakeResponse:
            raise urllib.error.URLError("down")

        confirmer = self.make(monkeypatch, fake_urlopen)

        assert confirmer.poll_decision("42") is None

    def test_mark_booked_edits_content_and_swallows_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> FakeResponse:
            raise http_error(404)

        confirmer = self.make(monkeypatch, fake_urlopen)

        confirmer.mark_booked("42", "note")  # must not raise
        confirmer.mark_declined("42")
        confirmer.mark_expired("42")


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
