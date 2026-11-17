"""Learned preferences, piece 1: arrivals become labelled training examples."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app import learning
from app.db import session_scope
from app.db.models import PreferenceSample
from app.events import Event, EventType, bus
from app.learning.samples import describe, features_at
from tests.test_platform import _setup_house, client  # noqa: F401  (fixture)


def _arrive(person_id: int, name: str = "Sam") -> None:
    asyncio.run(bus.publish(Event(type=EventType.PRESENCE_ARRIVED, source="vision:front_door",
                                  person_id=person_id, payload={"name": name})))


def _leave(person_id: int, name: str = "Sam") -> None:
    asyncio.run(bus.publish(Event(type=EventType.PRESENCE_LEFT, source="vision:front_door",
                                  person_id=person_id, payload={"name": name})))


def _samples() -> list[PreferenceSample]:
    with session_scope() as s:
        return s.query(PreferenceSample).order_by(PreferenceSample.id).all()


# ---------------------------------------------------------------- settings


@pytest.mark.parametrize("kind,state,setting", [
    ("light", {"state": "OFF", "brightness": 80}, "OFF"),
    ("light", {"state": "ON", "brightness": 25}, "ON:30"),
    ("light", {"state": "ON", "brightness": 60}, "ON:60"),
    ("light", {"state": "ON", "brightness": 90}, "ON:100"),
    ("switch", {"state": "ON"}, "ON"),
    ("thermostat", {"mode": "off", "target": 70}, "off"),
    ("thermostat", {"mode": "cool", "target": 70}, "cool:70"),
])
def test_settings_round_trip(kind, state, setting):
    assert learning.setting_of(kind, state) == setting
    # Applying the setting's command lands back in the same setting.
    assert learning.setting_of(kind, {**state, **learning.payload_for(kind, setting)}) == setting


def test_descriptions_read_naturally():
    assert describe("light", "ON:60") == "on at 60%"
    assert describe("thermostat", "cool:70") == "cool to 70°F"
    assert describe("switch", learning.LEAVE) == "left as it was"


def test_locks_and_garage_are_never_learnable():
    assert "lock" not in learning.LEARNABLE_KINDS and "garage" not in learning.LEARNABLE_KINDS
    with pytest.raises(ValueError):
        learning.setting_of("lock", {"state": "LOCKED"})


def test_features():
    f = features_at(datetime(2026, 11, 14, 18, 30), 80.4, 1)     # a Saturday
    assert f == {"hour": 18.5, "weekday": 5, "weekend": 1, "temp_f": 80.4, "others_home": 1}
    assert features_at(datetime(2026, 11, 16, 8, 0), None, 0)["temp_f"] == 72.0


# ---------------------------------------------------------------- recording


def test_arrival_then_changes_become_labels(client):
    ids = _setup_house(client)
    _arrive(ids["sam"])

    [sample] = _samples()
    assert sample.labels is None                                  # window still open
    assert set(sample.before) == {"light.entry", "light.living_room", "thermostat.main", "switch.tv"}
    assert sample.features["temp_f"] > 0

    h = ids["admin"]
    client.post("/api/devices/light.living_room/command", json={"payload": {"brightness": 60}}, headers=h)
    client.post("/api/devices/thermostat.main/command",
                json={"payload": {"mode": "cool", "target": 70}}, headers=h)

    assert learning.finalize_due() == 0                           # too early
    later = datetime.now(timezone.utc) + timedelta(minutes=16)
    assert learning.finalize_due(later) == 1

    labels = _samples()[0].labels
    assert labels["light.living_room"] == "ON:60"
    assert labels["thermostat.main"] == "cool:70"
    assert labels["switch.tv"] == learning.LEAVE


def test_undoing_a_change_counts_as_leave(client):
    ids = _setup_house(client)
    _arrive(ids["sam"])
    h = ids["admin"]
    client.post("/api/devices/switch.tv/command", json={"payload": {"state": "ON"}}, headers=h)
    client.post("/api/devices/switch.tv/command", json={"payload": {"state": "OFF"}}, headers=h)
    learning.finalize_due(datetime.now(timezone.utc) + timedelta(hours=1))
    assert _samples()[0].labels["switch.tv"] == learning.LEAVE


def test_leaving_closes_the_sample_before_the_away_rule(client):
    ids = _setup_house(client)
    _arrive(ids["sam"])
    client.post("/api/devices/switch.tv/command", json={"payload": {"state": "ON"}}, headers=ids["admin"])

    _leave(ids["sam"])                                            # nobody home -> away shutdown
    state = {d["id"]: d["state"] for d in client.get("/api/devices").json()}
    assert state["switch.tv"]["state"] == "OFF"                   # the away rule ran...
    assert _samples()[0].labels["switch.tv"] == "ON"              # ...but the label is what Sam wanted


def test_samples_endpoint(client):
    ids = _setup_house(client)
    _arrive(ids["sam"])
    client.post("/api/devices/switch.tv/command", json={"payload": {"state": "ON"}}, headers=ids["admin"])
    learning.finalize_due(datetime.now(timezone.utc) + timedelta(hours=1))
    _arrive(ids["sam"])

    body = client.get("/api/learning/samples").json()
    assert body["totals"] == {"Sam": {"complete": 1, "open": 1}}
    # The welcome rule's lights stayed on, so they count as wanted too.
    assert body["recent"][1]["changed"]["TV"] == "on"
