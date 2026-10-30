"""The system's nervous system.

Everything that happens in the house -- a person detected, a sensor reading, a
device state change -- becomes an Event and is published on this bus. The vision
pipeline, the MQTT client, and the API are all producers; the rules engine, the
database logger, and the dashboard WebSocket are all consumers.

Keeping this in the middle is what makes the project a *platform* rather than a
pile of scripts: new producers and consumers plug in without touching each other.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)


class EventType(str, Enum):
    PERSON_DETECTED = "person_detected"
    PERSON_RECOGNIZED = "person_recognized"
    PERSON_SEEN = "person_seen"          # periodic "still in frame" for a recognized track
    UNKNOWN_PERSON = "unknown_person"
    PRESENCE_ARRIVED = "presence_arrived"
    PRESENCE_LEFT = "presence_left"
    SENSOR_READING = "sensor_reading"
    DEVICE_STATE = "device_state"
    DEVICE_COMMAND = "device_command"
    ACCESS_GRANTED = "access_granted"    # camera-verified person allowed to use a device
    ACCESS_DENIED = "access_denied"      # ...or refused, by permission or no face
    ADMIN_LOGIN = "admin_login"
    SYSTEM = "system"


@dataclass
class Event:
    type: EventType
    source: str                      # "vision:front_door", "mqtt:sensor/kitchen", "api"
    payload: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    person_id: int | None = None
    device_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "source": self.source,
            "payload": self.payload,
            "confidence": self.confidence,
            "person_id": self.person_id,
            "device_id": self.device_id,
            "timestamp": self.timestamp.isoformat(),
        }


Handler = Callable[[Event], Awaitable[None]]


class EventBus:
    """Minimal async pub/sub. One process, many subscribers, no external broker."""

    def __init__(self) -> None:
        self._handlers: list[tuple[set[EventType] | None, Handler]] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Remember the main loop so background threads can publish into it."""
        self._loop = loop

    def subscribe(self, handler: Handler, types: set[EventType] | None = None) -> None:
        self._handlers.append((types, handler))

    async def publish(self, event: Event) -> None:
        for types, handler in self._handlers:
            if types is not None and event.type not in types:
                continue
            try:
                await handler(event)
            except Exception:  # one bad subscriber must not kill the bus
                log.exception("event handler %s failed on %s", handler, event.type)

    def publish_threadsafe(self, event: Event) -> None:
        """Publish from a non-async thread (vision loop, paho callback)."""
        if self._loop is None:
            log.warning("event bus has no loop bound; dropping %s", event.type)
            return
        asyncio.run_coroutine_threadsafe(self.publish(event), self._loop)


bus = EventBus()
