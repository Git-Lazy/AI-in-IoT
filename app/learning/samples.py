"""Turning arrivals into training examples.

When someone arrives we write down the situation (features) and every learnable
device's setting (before). When the learning window closes -- or they leave
first -- we look again: a device whose setting changed gets that setting as its
label, an untouched one gets "leave".

Labelling by the end result rather than by who pressed what means the model's
own suggestions are judged too: if it turned the light on and the person left
it on, that counts as "wanted ON"; if they switched it back off, the light
ended where it started and the label is "leave".

Settings are coarse on purpose (three brightness levels, AC targets to the
degree) so a few weeks of arrivals are enough to learn from.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.config import settings
from app.db import session_scope
from app.db.models import Device, Person, PreferenceSample
from app.events import Event

log = logging.getLogger(__name__)

# Locks and the garage are never learned or automated: a wrong guess there
# isn't an inconvenience, it's an open front door.
LEARNABLE_KINDS = ("light", "switch", "thermostat")
LEAVE = "leave"
DEFAULT_TEMP_F = 72.0

# Brightness buckets: (upper bound, level we set).
_BRIGHTNESS = ((40, 30), (75, 60), (100, 100))


# --------------------------------------------------------------- settings


def setting_of(kind: str, state: dict[str, Any]) -> str:
    """A device's state as one of a small set of named settings."""
    if kind == "light":
        if str(state.get("state", "OFF")).upper() != "ON":
            return "OFF"
        level = int(state.get("brightness", 100))
        bucket = next(b for limit, b in _BRIGHTNESS if level <= limit)
        return f"ON:{bucket}"
    if kind == "switch":
        return "ON" if str(state.get("state", "OFF")).upper() == "ON" else "OFF"
    if kind == "thermostat":
        mode = str(state.get("mode", "off")).lower()
        return "off" if mode == "off" else f"{mode}:{int(state.get('target', 72))}"
    raise ValueError(f"{kind} isn't learnable")


def payload_for(kind: str, setting: str) -> dict[str, Any]:
    """The command that puts a device into a named setting."""
    if kind == "light":
        if setting == "OFF":
            return {"state": "OFF"}
        return {"state": "ON", "brightness": int(setting.split(":")[1])}
    if kind == "switch":
        return {"state": setting}
    if kind == "thermostat":
        if setting == "off":
            return {"mode": "off"}
        mode, target = setting.split(":")
        return {"mode": mode, "target": int(target)}
    raise ValueError(f"{kind} isn't learnable")


def describe(kind: str, setting: str) -> str:
    """Human wording: "ON:60" -> "on at 60%", "cool:70" -> "cool to 70°F"."""
    if setting == LEAVE:
        return "left as it was"
    if kind == "light" and setting.startswith("ON:"):
        return f"on at {setting.split(':')[1]}%"
    if kind == "thermostat" and setting != "off":
        mode, target = setting.split(":")
        return f"{mode} to {target}°F"
    return setting.lower()


# --------------------------------------------------------------- features


def features_at(when: datetime, temp_f: float | None, others_home: int) -> dict[str, float]:
    """The situation as numbers a model can split on. `when` in local time."""
    return {
        "hour": round(when.hour + when.minute / 60, 2),
        "weekday": when.weekday(),                 # 0 = Monday
        "weekend": int(when.weekday() >= 5),
        "temp_f": round(temp_f if temp_f is not None else DEFAULT_TEMP_F, 1),
        "others_home": others_home,
    }


def _indoor_temp(session) -> float | None:
    sensor = session.query(Device).filter(Device.kind == "temperature_sensor").first()
    if sensor is None:
        return None
    try:
        return float((sensor.state or {}).get("temperature"))
    except (TypeError, ValueError):
        return None


def _settings_now(session) -> dict[str, str]:
    out = {}
    for device in session.query(Device).filter(Device.kind.in_(LEARNABLE_KINDS)):
        try:
            out[device.id] = setting_of(device.kind, device.state or {})
        except (TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------- recording


async def on_arrival(event: Event) -> None:
    """PRESENCE_ARRIVED: open a sample. Runs before the rules change anything."""
    if event.person_id is None:
        return
    with session_scope() as session:
        if session.get(Person, event.person_id) is None:
            return
        others = (
            session.query(Person)
            .filter(Person.is_home.is_(True), Person.id != event.person_id)
            .count()
        )
        when = event.timestamp.astimezone()          # local time: habits follow the clock on the wall
        session.add(PreferenceSample(
            person_id=event.person_id,
            arrived_at=event.timestamp,
            features=features_at(when, _indoor_temp(session), others),
            before=_settings_now(session),
        ))
    log.info("learning: recorded arrival of person %s", event.person_id)


async def on_left(event: Event) -> None:
    """PRESENCE_LEFT: close their open samples now, before the away rule
    switches everything off (that's not what they wanted, it's them leaving)."""
    if event.person_id is None:
        return
    with session_scope() as session:
        _close(session, session.query(PreferenceSample).filter(
            PreferenceSample.person_id == event.person_id,
            PreferenceSample.labels.is_(None),
        ).all())


def finalize_due(now: datetime | None = None) -> int:
    """Close every sample whose learning window has passed. Returns how many."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=settings.learning_window_minutes)
    with session_scope() as session:
        due = [
            s for s in session.query(PreferenceSample).filter(PreferenceSample.labels.is_(None))
            if _aware(s.arrived_at) <= cutoff
        ]
        return _close(session, due)


def _close(session, samples: list[PreferenceSample]) -> int:
    if not samples:
        return 0
    after = _settings_now(session)
    for sample in samples:
        sample.labels = {
            device_id: (after[device_id] if after[device_id] != before else LEAVE)
            for device_id, before in (sample.before or {}).items()
            if device_id in after
        }
        changed = {d: s for d, s in sample.labels.items() if s != LEAVE}
        log.info("learning: sample %s closed, changed %s", sample.id, changed or "nothing")
    return len(samples)


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)      # SQLite drops tzinfo
