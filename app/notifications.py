"""Notification delivery (MVP #7).

Every notification lands in an in-memory ring buffer the dashboard polls, and --
when NTFY_TOPIC is set -- is pushed to your phone through ntfy (https://ntfy.sh):
install the ntfy app, subscribe to the same topic, done. No account needed.

Pushing never blocks the rules engine: it runs as a background task, and a
failed push is logged, not raised. Repeats of the same alert inside
NTFY_COOLDOWN_SECONDS are kept on the dashboard but not pushed again, so someone
lingering at the door doesn't buzz your phone every few seconds.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

import httpx

from app.config import settings

log = logging.getLogger(__name__)

PRIORITIES = ("low", "normal", "high")
# ntfy priority 1-5; 5 overrides Do Not Disturb on most phones.
_NTFY_PRIORITY = {"low": "2", "normal": "3", "high": "5"}
_NTFY_TAGS = {"low": "house", "normal": "house", "high": "rotating_light"}
PUSH_TIMEOUT_SECONDS = 10


class Notifier:
    def __init__(self, keep: int = 100) -> None:
        self._recent: deque[dict[str, Any]] = deque(maxlen=keep)
        self._last_push: dict[str, float] = {}
        self._tasks: set[asyncio.Task] = set()

    @property
    def phone_enabled(self) -> bool:
        return bool(settings.ntfy_topic.strip())

    def send(
        self,
        message: str,
        priority: str = "normal",
        *,
        title: str = "Home AI",
        image: bytes | None = None,
        key: str | None = None,
    ) -> None:
        """Record a notification and push it to the phone if that's configured.

        `key` groups repeats for the cooldown (defaults to the message text);
        `image` is a JPEG attached to the push, e.g. a camera snapshot.
        """
        if priority not in PRIORITIES:
            priority = "normal"
        entry = {
            "message": message,
            "priority": priority,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pushed": False,
        }
        self._recent.appendleft(entry)
        log.info("[notify:%s] %s", priority, message)

        if self._should_push(priority, key or message):
            entry["pushed"] = True
            self._spawn(self.push(message, priority, title=title, image=image))

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return list(self._recent)[:limit]

    # ------------------------------------------------------------------ phone

    def _should_push(self, priority: str, key: str) -> bool:
        if not self.phone_enabled:
            return False
        if PRIORITIES.index(priority) < PRIORITIES.index(settings.ntfy_min_priority):
            return False
        now = time.monotonic()
        last = self._last_push.get(key)
        if last is not None and now - last < settings.ntfy_cooldown_seconds:
            log.debug("push skipped (cooldown): %s", key)
            return False
        self._last_push[key] = now
        return True

    def _spawn(self, coro) -> None:
        try:
            task = asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            # Called from a camera thread or a script: no loop here, use our own.
            threading.Thread(target=asyncio.run, args=(coro,), daemon=True).start()
            return
        self._tasks.add(task)              # keep a reference until it finishes
        task.add_done_callback(self._tasks.discard)

    async def push(
        self, message: str, priority: str = "normal", *, title: str = "Home AI",
        image: bytes | None = None,
    ) -> bool:
        """Deliver one notification to ntfy. Returns True on success."""
        url = f"{settings.ntfy_server.rstrip('/')}/{settings.ntfy_topic.strip()}"
        headers = {
            "Title": _header(title),
            "Priority": _NTFY_PRIORITY.get(priority, "3"),
            "Tags": _NTFY_TAGS.get(priority, "house"),
        }
        if settings.ntfy_token:
            headers["Authorization"] = f"Bearer {settings.ntfy_token}"
        if image is not None:
            # With an attachment the body is the file and the text moves to a header.
            headers["Message"] = _header(message)
            headers["Filename"] = "snapshot.jpg"
            body: bytes = image
        else:
            body = message.encode("utf-8")

        try:
            async with httpx.AsyncClient(timeout=PUSH_TIMEOUT_SECONDS) as client:
                response = await client.put(url, content=body, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("phone push failed: %s", exc)
            return False
        log.info("pushed to phone (ntfy topic %s)", settings.ntfy_topic)
        return True


def _header(text: str) -> str:
    """HTTP headers are ASCII-only; ntfy decodes RFC 2047 for anything else."""
    if text.isascii():
        return text
    return "=?UTF-8?B?" + base64.b64encode(text.encode("utf-8")).decode("ascii") + "?="


notifier = Notifier()
