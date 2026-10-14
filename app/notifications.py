"""Notification delivery (MVP #7).

Starts as a log sink plus an in-memory ring buffer the dashboard polls. Swapping
in ntfy/Pushover/email later means implementing one `send` method -- callers in
the rules engine don't change.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)


class Notifier:
    def __init__(self, keep: int = 100) -> None:
        self._recent: deque[dict[str, Any]] = deque(maxlen=keep)

    def send(self, message: str, priority: str = "normal") -> None:
        entry = {
            "message": message,
            "priority": priority,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._recent.appendleft(entry)
        log.info("[notify:%s] %s", priority, message)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return list(self._recent)[:limit]


notifier = Notifier()
