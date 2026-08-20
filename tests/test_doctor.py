"""Tests for the `midas doctor` Discord diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from midas.alerts import DiscordApiError
from midas.doctor import run_discord_checks
from midas.models import AlertsConfig


class FakeClient:
    """DiscordBotClient double with per-method scripted failures."""

    def __init__(self, fail: dict[str, DiscordApiError] | None = None) -> None:
        self.fail = fail or {}
        self.calls: list[str] = []
        self.edits: list[dict[str, Any]] = []
        self.posted: list[dict[str, Any]] = []

    def _maybe_fail(self, method: str) -> None:
        self.calls.append(method)
        if method in self.fail:
            raise self.fail[method]

    def get_current_user(self) -> dict[str, Any]:
        self._maybe_fail("get_current_user")
        return {"id": "1", "username": "midas-bot"}

    def get_channel(self, channel_id: str) -> dict[str, Any]:
        self._maybe_fail(f"get_channel:{channel_id}")
        return {"id": channel_id, "name": "chan"}

    def create_message(self, channel_id: str, payload: dict[str, Any]) -> str:
        self._maybe_fail("create_message")
        self.posted.append(payload)
        return "42"

    def get_message(self, channel_id: str, message_id: str) -> dict[str, Any]:
        self._maybe_fail("get_message")
        return {"id": message_id, "reactions": []}

    def edit_message(self, channel_id: str, message_id: str, payload: dict[str, Any]) -> None:
        self._maybe_fail("edit_message")
        self.edits.append(payload)


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    client = FakeClient()
    monkeypatch.setattr("midas.doctor.DiscordBotClient", lambda *a, **k: client)
    return client


def run(
    tmp_path: Path,
    config: AlertsConfig | None,
    token: str = "tok",
    wait_reaction: bool = False,
) -> tuple[Any, list[str]]:
    lines: list[str] = []
    result = run_discord_checks(
        config,
        token,
        tmp_path / "state.yaml",
        provider=None,  # type: ignore[arg-type] — snapshot degrades to sample data
        echo=lines.append,
        wait_reaction=wait_reaction,
    )
    return result, lines


def test_unconfigured_is_healthy_noop(tmp_path: Path) -> None:
    result, lines = run(tmp_path, None, token="")
    assert result.healthy
    assert any("not configured" in line for line in lines)


def test_channel_without_token_fails(tmp_path: Path) -> None:
    result, lines = run(tmp_path, AlertsConfig(discord_intra_day_channel_id="1"), token="")
    assert not result.healthy
    assert any("MIDAS_DISCORD_BOT_TOKEN" in line for line in lines)


def test_token_without_channel_fails(tmp_path: Path) -> None:
    result, _ = run(tmp_path, AlertsConfig())
    assert not result.healthy


def test_full_healthy_run_posts_and_edits(tmp_path: Path, fake_client: FakeClient) -> None:
    config = AlertsConfig(discord_intra_day_channel_id="111", discord_end_of_day_channel_id="222")
    result, lines = run(tmp_path, config)

    assert result.healthy
    # Intra-day: test order + edit-to-complete; end-of-day: test report.
    assert len(fake_client.posted) == 2
    assert fake_client.posted[0]["embeds"][0]["title"].startswith("🧪 TEST")
    assert fake_client.posted[1]["embeds"][0]["title"].startswith("🧪 TEST report")
    assert fake_client.edits and "Test complete" in fake_client.edits[0]["content"]
    # Sample-data fallback provenance is surfaced (no real state/provider here).
    assert any("sample numbers" in line for line in lines)


def test_bad_token_fails_with_hint(tmp_path: Path, fake_client: FakeClient) -> None:
    fake_client.fail["get_current_user"] = DiscordApiError("HTTP 401", status=401)
    result, lines = run(tmp_path, AlertsConfig(discord_intra_day_channel_id="111"))

    assert not result.healthy
    assert any("token is invalid" in line for line in lines)


def test_invisible_channel_fails_with_invite_hint(tmp_path: Path, fake_client: FakeClient) -> None:
    fake_client.fail["get_channel:111"] = DiscordApiError("HTTP 403", status=403)
    config = AlertsConfig(discord_intra_day_channel_id="111", discord_end_of_day_channel_id="222")
    result, lines = run(tmp_path, config)

    assert not result.healthy
    assert any("invite it" in line for line in lines)
    # The end-of-day channel is still checked despite the intra-day failure.
    assert "get_channel:222" in fake_client.calls


def test_readback_failure_flags_polling_blindness(tmp_path: Path, fake_client: FakeClient) -> None:
    fake_client.fail["get_message"] = DiscordApiError("HTTP 403", status=403)
    result, lines = run(tmp_path, AlertsConfig(discord_intra_day_channel_id="111"))

    assert not result.healthy
    assert any("Read Message History" in line for line in lines)


def test_wait_reaction_sees_operator_tap(tmp_path: Path, fake_client: FakeClient) -> None:
    fake_client.get_message = lambda channel_id, message_id: {  # type: ignore[method-assign]
        "id": message_id,
        "reactions": [{"emoji": {"name": "✅"}, "count": 1}],
    }
    result, lines = run(tmp_path, AlertsConfig(discord_intra_day_channel_id="111"), wait_reaction=True)

    assert result.healthy
    assert any("round-trip verified" in line for line in lines)
