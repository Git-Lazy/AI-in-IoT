"""Generated arrival history, so there's something to learn from on day one.

A real household produces one training example per arrival -- a few weeks of
living with the system before it has much to go on. For development and the
demo, this writes weeks of arrivals that follow a set of hidden habits (below),
with some randomness, so we can check the model recovers the habits on its own.

Generated rows are marked source="demo" and can be removed in one go; real
arrivals are never touched.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from app.db import session_scope
from app.db.models import Device, PreferenceSample
from app.learning.samples import LEARNABLE_KINDS, LEAVE, features_at, setting_of

# How often the generated person does something other than their habit.
NOISE = 0.1

# The hidden habits. Each gets (features, device kind, device room) and returns
# a setting or LEAVE. These are what the model should rediscover.
#   * Living-room light: evenings dim on weekends (movie), medium on weekdays.
#   * Entry light: full brightness when arriving after dark.
#   * AC: cool to 70 when it's warm inside, heat to 70 when it's cold.
#   * TV: on when arriving on the weekend, or on weekday evenings after 7.


def habit(f: dict, kind: str, room: str) -> str:
    evening = f["hour"] >= 18
    if kind == "light" and room == "Living Room":
        if not evening:
            return LEAVE
        return "ON:30" if f["weekend"] else "ON:60"
    if kind == "light" and room == "Entry":
        return "ON:100" if evening else LEAVE
    if kind == "thermostat":
        if f["temp_f"] >= 78:
            return "cool:70"
        if f["temp_f"] <= 65:
            return "heat:70"
        return LEAVE
    if kind == "switch":
        return "ON" if f["weekend"] or f["hour"] >= 19 else LEAVE
    return LEAVE


def _arrival_times(rng: random.Random, weeks: int, now: datetime) -> list[datetime]:
    """About one arrival a day: weekday evenings, weekends any time."""
    out = []
    start = (now - timedelta(weeks=weeks)).replace(hour=0, minute=0, second=0, microsecond=0)
    for day in range(weeks * 7):
        date = start + timedelta(days=day)
        weekend = date.weekday() >= 5
        for _ in range(rng.choice((1, 1, 1, 2)) if weekend else rng.choice((1, 1, 1, 1, 0))):
            hour = rng.uniform(10, 21) if weekend else rng.gauss(18, 1.5)
            hour = min(max(hour, 7), 23.5)
            when = date + timedelta(hours=hour)
            if when < now:
                out.append(when)
    return out


def seed(person_id: int, weeks: int = 6, rng_seed: int = 7) -> int:
    """Write `weeks` of generated arrivals for one person. Returns how many."""
    rng = random.Random(rng_seed)
    now = datetime.now().astimezone()
    with session_scope() as session:
        devices = [
            (d.id, d.kind, d.room)
            for d in session.query(Device).filter(Device.kind.in_(LEARNABLE_KINDS))
        ]
        if not devices:
            raise LookupError("no lights, plugs or AC registered -- add the demo devices first")
        rows = 0
        for when in _arrival_times(rng, weeks, now):
            # Indoor temperature: warmer in the afternoon, plus a hot or cold spell now and then.
            temp = rng.gauss(72, 6) + (3 if 12 <= when.hour <= 18 else 0)
            features = features_at(when, temp, int(rng.random() < 0.25))
            labels = {}
            before = {}
            for device_id, kind, room in devices:
                before[device_id] = setting_of(kind, {})       # coming home to a house that's off
                label = habit(features, kind, room)
                if rng.random() < NOISE:
                    label = rng.choice([LEAVE, label, _other(kind, rng)])
                labels[device_id] = label
            session.add(PreferenceSample(
                person_id=person_id, arrived_at=when.astimezone(timezone.utc),
                features=features, before=before, labels=labels, source="demo",
            ))
            rows += 1
    return rows


def _other(kind: str, rng: random.Random) -> str:
    return rng.choice({
        "light": ["ON:30", "ON:60", "ON:100"],
        "switch": ["ON"],
        "thermostat": ["cool:68", "cool:72", "heat:70"],
    }[kind])


def clear() -> int:
    """Remove every generated arrival. Real ones stay."""
    with session_scope() as session:
        return session.query(PreferenceSample).filter(PreferenceSample.source == "demo").delete()
