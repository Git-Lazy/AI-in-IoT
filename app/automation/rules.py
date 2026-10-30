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
from typing import Any, Awaitable, Callable

from app import access
from app.db import session_scope
from app.db.models import Device, EventLog, Person, Preference
from app.events import Event, EventType, bus
from app.iot import devices, mqtt_bridge
from app.notifications import notifier

log = logging.getLogger(__name__)


@dataclass
class Rule:
    name: str
    when: EventType
    condition: Callable[[Event], bool]
    action: Callable[[Event], Awaitable[None]]
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
    """Keep the devices table current from MQTT and simulator state reports."""
    if event.device_id is None:
        return
    state = {k: v for k, v in event.payload.items() if k != "kind"}
    with session_scope() as session:
        device = session.get(Device, event.device_id)
        if device is None:
            # An unknown device announced itself over MQTT: register it as real hardware.
            device = Device(
                id=event.device_id,
                name=event.device_id,
                kind=event.payload.get("kind", "sensor"),
                transport="mqtt",
            )
            session.add(device)
        device.state = state
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


def _camera_room(event: Event) -> str:
    """Room of the camera that produced a vision event ("vision:<camera>")."""
    from app.vision import cameras

    camera = event.payload.get("camera") or event.source.removeprefix("vision:")
    return cameras.room_of(camera)


async def _command(device_id: str, payload: dict[str, Any], source: str, actor: str | None = None) -> bool:
    try:
        await devices.send(device_id, payload, source=source, actor=actor)
        return True
    except (LookupError, devices.CommandError) as exc:
        log.warning("automation skipped %s: %s", device_id, exc)
        return False


def _device_ids(kinds: tuple[str, ...], room: str | None = None) -> list[str]:
    with session_scope() as session:
        query = session.query(Device).filter(Device.kind.in_(kinds))
        if room:
            query = query.filter(Device.room == room)
        return [d.id for d in query]


async def welcome_home(event: Event) -> None:
    """Apply the arriving person's preferences -- only what they're allowed to use.

    No preferences saved: turn on the lights in the room where they were seen.
    """
    with session_scope() as session:
        person = session.get(Person, event.person_id)
        if person is None:
            return
        name = person.name
        prefs = session.query(Preference).filter(Preference.person_id == person.id).all()
        plan: list[tuple[str, dict]] = []
        skipped: list[str] = []
        if prefs:
            targets = [(p.device_id, p.payload) for p in prefs]
        else:
            room = _camera_room(event)
            targets = [(d, {"state": "ON"}) for d in _device_ids(("light",), room)]
        for device_id, payload in targets:
            device = session.get(Device, device_id)
            if device is None:
                continue
            allowed, _ = access.can_control(session, person, device)
            if allowed:
                plan.append((device_id, payload))
            else:
                skipped.append(device.name)

    done = 0
    for device_id, payload in plan:
        if await _command(device_id, payload, "automation:arrival", name):
            done += 1
    message = f"Welcome home, {name}."
    if done:
        message += f" Set {done} device(s) to your preferences."
    if skipped:
        message += f" Not allowed: {', '.join(skipped)}."
    notifier.send(message)


async def away_shutdown(event: Event) -> None:
    """Once the last person has left, switch off lights and plugs and the AC."""
    name = event.payload.get("name", "Someone")
    with session_scope() as session:
        anyone_home = session.query(Person).filter(Person.is_home.is_(True)).count() > 0
    if anyone_home:
        notifier.send(f"{name} left the house.")
        return

    for device_id in _device_ids(("light", "switch")):
        await _command(device_id, {"state": "OFF"}, "automation:away")
    for device_id in _device_ids(("thermostat",)):
        await _command(device_id, {"mode": "off"}, "automation:away")
    notifier.send(f"{name} left -- nobody is home. Lights and AC are off.")


async def unknown_at_door(event: Event) -> None:
    """Light up the room an unrecognized person is in, and raise an alert."""
    room = _camera_room(event)
    for device_id in _device_ids(("light",), room):
        await _command(device_id, {"state": "ON", "brightness": 100}, "automation:security")
    notifier.send(
        f"Unrecognized person at {event.payload.get('camera', 'a camera')}.",
        priority="high",
    )


RULES: list[Rule] = [
    Rule(
        name="welcome_home",
        when=EventType.PRESENCE_ARRIVED,
        condition=lambda e: e.person_id is not None,
        action=welcome_home,
        description="Apply the arriving person's saved preferences (MVP #9).",
    ),
    Rule(
        name="away_shutdown",
        when=EventType.PRESENCE_LEFT,
        condition=lambda e: True,
        action=away_shutdown,
        description="Power down lights and AC once the last person leaves.",
    ),
    Rule(
        name="unknown_at_door",
        when=EventType.UNKNOWN_PERSON,
        condition=lambda e: True,
        action=unknown_at_door,
        description="Light the room and alert on an unrecognized face (MVP #7).",
    ),
]


async def evaluate_rules(event: Event) -> None:
    for rule in RULES:
        if rule.when != event.type:
            continue
        try:
            if rule.condition(event):
                log.info("rule fired: %s", rule.name)
                await rule.action(event)
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
