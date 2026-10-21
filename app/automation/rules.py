"""The decision layer: events in, device commands and notifications out.

This is where the Capstone's central claim lives -- that an AI layer sitting
between vision and IoT can combine both into decisions neither could make alone.
Right now the rules are explicit and declarative. Weeks 6-7 extend this with
dwell-time logic, confidence gating, and LLM-assisted interpretation of ambiguous
events; keeping rules as data makes that extension additive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from app.db import session_scope
from app.db.models import Device, EventLog, Person
from app.events import Event, EventType, bus
from app.iot import mqtt_bridge
from app.notifications import notifier

log = logging.getLogger(__name__)


@dataclass
class Rule:
    name: str
    when: EventType
    condition: Callable[[Event], bool]
    action: Callable[[Event], None]
    description: str = ""


# ----------------------------------------------------------------- persistence


async def log_event(event: Event) -> None:
    """Every event on the bus gets written to the database (MVP #6)."""
    with session_scope() as session:
        session.add(
            EventLog(
                type=event.type.value,
                source=event.source,
                payload=event.payload,
                confidence=event.confidence,
                person_id=event.person_id,
                device_id=event.device_id,
                timestamp=event.timestamp,
            )
        )


async def sync_device_state(event: Event) -> None:
    """Keep the devices table current from MQTT state messages."""
    if event.device_id is None:
        return
    with session_scope() as session:
        device = session.get(Device, event.device_id)
        if device is None:
            device = Device(
                id=event.device_id,
                name=event.device_id,
                kind=event.payload.get("kind", "sensor"),
            )
            session.add(device)
        device.state = event.payload
        device.online = True
        device.last_update = event.timestamp


async def mirror_to_mqtt(event: Event) -> None:
    """Republish vision events so Home Assistant can subscribe to them."""
    if event.source.startswith("vision:"):
        mqtt_bridge.announce(event)


# ------------------------------------------------------------ presence tracking

# How long without a sighting before we call someone "left".
PRESENCE_TIMEOUT = timedelta(minutes=5)


async def track_presence(event: Event) -> None:
    """Turn repeated recognitions into arrive/leave transitions (MVP #5).

    A single recognition is noisy; a transition is meaningful. Debouncing here is
    the main defense against the false-positive problem named in the proposal.
    """
    if event.person_id is None:
        return

    with session_scope() as session:
        person = session.get(Person, event.person_id)
        if person is None:
            return

        now = event.timestamp
        was_home = person.is_home
        person.last_seen = now
        person.is_home = True
        name = person.name

    if not was_home:
        await bus.publish(
            Event(
                type=EventType.PRESENCE_ARRIVED,
                source=event.source,
                person_id=event.person_id,
                payload={"name": name},
            )
        )


async def expire_presence() -> None:
    """Mark people as gone once they haven't been seen for PRESENCE_TIMEOUT.

    Called on a timer by the scheduler in main.py.
    """
    cutoff = datetime.now(timezone.utc) - PRESENCE_TIMEOUT
    departed: list[tuple[int, str]] = []

    with session_scope() as session:
        for person in session.query(Person).filter(Person.is_home.is_(True)).all():
            last = person.last_seen
            if last is None:
                continue
            if last.tzinfo is None:                     # SQLite drops tzinfo
                last = last.replace(tzinfo=timezone.utc)
            if last < cutoff:
                person.is_home = False
                departed.append((person.id, person.name))

    for person_id, name in departed:
        await bus.publish(
            Event(
                type=EventType.PRESENCE_LEFT,
                source="presence:timeout",
                person_id=person_id,
                payload={"name": name},
            )
        )


# ------------------------------------------------------------------ the rules


def _command(device_id: str, payload: dict[str, Any]) -> None:
    mqtt_bridge.command(device_id, payload)


RULES: list[Rule] = [
    Rule(
        name="welcome_home",
        when=EventType.PRESENCE_ARRIVED,
        condition=lambda e: True,
        action=lambda e: (
            _command("light.entry", {"state": "ON", "brightness": 180}),
            notifier.send(f"Welcome home, {e.payload.get('name', 'someone')}."),
        ),
        description="Turn on the entry light when a household member arrives (MVP #9).",
    ),
    Rule(
        name="away_shutdown",
        when=EventType.PRESENCE_LEFT,
        condition=lambda e: True,
        action=lambda e: (
            _command("light.entry", {"state": "OFF"}),
            notifier.send(f"{e.payload.get('name', 'Someone')} left the house."),
        ),
        description="Power down the entry light once the last person leaves.",
    ),
    Rule(
        name="unknown_at_door",
        when=EventType.UNKNOWN_PERSON,
        condition=lambda e: (e.confidence or 0) >= 0.0,
        action=lambda e: (
            _command("light.porch", {"state": "ON"}),
            notifier.send(
                f"Unrecognized person at {e.payload.get('camera', 'a camera')}.",
                priority="high",
            ),
        ),
        description="Light the porch and alert on an unrecognized face (MVP #7).",
    ),
]


async def evaluate_rules(event: Event) -> None:
    for rule in RULES:
        if rule.when != event.type:
            continue
        try:
            if rule.condition(event):
                log.info("rule fired: %s", rule.name)
                rule.action(event)
        except Exception:
            log.exception("rule %s failed", rule.name)


# ---------------------------------------------------------------- registration


def register_handlers() -> None:
    """Wire every consumer onto the bus. Called once at startup."""
    bus.subscribe(log_event)
    bus.subscribe(sync_device_state, {EventType.DEVICE_STATE, EventType.SENSOR_READING})
    bus.subscribe(mirror_to_mqtt)
    bus.subscribe(track_presence, {EventType.PERSON_RECOGNIZED, EventType.PERSON_SEEN})
    bus.subscribe(evaluate_rules)
    log.info("automation handlers registered (%d rules)", len(RULES))
