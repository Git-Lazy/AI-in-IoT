"""Phone notifications: what gets pushed, how it's sent, and the cooldown."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app import notifications
from app.config import settings
from app.notifications import Notifier


@pytest.fixture
def ntfy(monkeypatch):
    """Capture every request the notifier makes instead of hitting ntfy.sh."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"id": "x"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        notifications.httpx, "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    monkeypatch.setattr(settings, "ntfy_topic", "homeai-test-topic")
    monkeypatch.setattr(settings, "ntfy_server", "https://ntfy.example")
    monkeypatch.setattr(settings, "ntfy_min_priority", "normal")
    monkeypatch.setattr(settings, "ntfy_cooldown_seconds", 60)
    return sent


def _send_all(notifier: Notifier, *calls: dict) -> None:
    async def run():
        for kw in calls:
            notifier.send(**kw)
        await asyncio.gather(*notifier._tasks)

    asyncio.run(run())


def test_push_goes_to_topic_with_priority(ntfy):
    n = Notifier()
    _send_all(n, {"message": "Welcome home, Sam.", "priority": "normal"})

    assert len(ntfy) == 1
    req = ntfy[0]
    assert str(req.url) == "https://ntfy.example/homeai-test-topic"
    assert req.headers["Priority"] == "3"
    assert req.content == b"Welcome home, Sam."
    assert n.recent()[0]["pushed"] is True


def test_snapshot_is_attached_and_text_moves_to_header(ntfy):
    _send_all(Notifier(), {"message": "Unrecognized person at front_door.", "priority": "high",
                           "image": b"\xff\xd8jpeg", "key": "unknown:front_door"})

    req = ntfy[0]
    assert req.content == b"\xff\xd8jpeg"
    assert req.headers["Filename"] == "snapshot.jpg"
    assert req.headers["Message"] == "Unrecognized person at front_door."
    assert req.headers["Priority"] == "5"


def test_cooldown_and_min_priority(ntfy):
    n = Notifier()
    _send_all(
        n,
        {"message": "Unknown at door", "priority": "high", "key": "unknown:front_door"},
        {"message": "Unknown at door", "priority": "high", "key": "unknown:front_door"},
        {"message": "Unknown at garage", "priority": "high", "key": "unknown:garage"},
        {"message": "just FYI", "priority": "low"},
    )
    # The repeat is held back, a different camera isn't, low stays on the dashboard.
    assert [r.headers["Message"] if "Message" in r.headers else r.content.decode() for r in ntfy] \
        == ["Unknown at door", "Unknown at garage"]
    assert len(n.recent()) == 4


def test_no_topic_means_dashboard_only(ntfy, monkeypatch):
    monkeypatch.setattr(settings, "ntfy_topic", "")
    n = Notifier()
    _send_all(n, {"message": "hello", "priority": "high"})
    assert ntfy == [] and n.recent()[0]["pushed"] is False


def test_non_ascii_title_is_encoded(ntfy):
    _send_all(Notifier(), {"message": "hi", "title": "Bienvenue, Zoë"})
    assert ntfy[0].headers["Title"].startswith("=?UTF-8?B?")


def test_failed_push_is_logged_not_raised(monkeypatch):
    def boom(request):
        raise httpx.ConnectError("offline")

    real_client = httpx.AsyncClient
    monkeypatch.setattr(notifications.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(boom), **kw))
    monkeypatch.setattr(settings, "ntfy_topic", "t")
    assert asyncio.run(Notifier().push("hello")) is False
