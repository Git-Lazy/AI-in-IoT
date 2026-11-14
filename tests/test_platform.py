"""Admin, permissions, camera-checked control, the simulator, and face voting."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import auth
from app.db import engine, init_db, session_scope
from app.db.models import Base, Device, FaceEmbedding, Person
from app.events import EventType
from app.iot import devices
from app.iot.simulator import Simulator, _step_thermostat
from app.vision import cameras
from app.vision.recognizer import FaceGallery, Match

# --------------------------------------------------------------- fixtures


class FakeCamera:
    """Stands in for a VisionPipeline: a room and whoever it 'sees'."""

    def __init__(self, name: str, room: str, present: list[Match] | None = None):
        self.camera_name, self.room, self.present = name, room, present or []
        self.door = None

    def set_door(self, line):
        self.door = line

    def present_people(self):
        return list(self.present)

    def reload_gallery(self):
        pass

    def stop(self):
        pass


class FakeRecognizer:
    model_name = "stub/haar"


@pytest.fixture
def client(monkeypatch):
    Base.metadata.drop_all(engine)
    init_db()
    auth._sessions.clear()
    monkeypatch.setattr("app.api.routes_devices.FACE_WAIT_SECONDS", 0.0)
    monkeypatch.setattr("app.api.routes_admin.FACE_WAIT_SECONDS", 0.3)
    # Windows takes ~2s to refuse a closed port; tests don't need a broker at all.
    monkeypatch.setattr("app.iot.mqtt_client.mqtt_bridge.start", lambda: None)
    # Each app startup subscribes the handlers again; without a fresh list every
    # event would be handled once per earlier test.
    from app.events import bus

    monkeypatch.setattr(bus, "_handlers", [])
    from app.main import app

    with TestClient(app) as c:
        cameras._pipelines = {}
        cameras.recognizer = None
        yield c
    cameras._pipelines = {}


def _setup_house(c: TestClient) -> dict:
    """Admin + child + the demo devices. Returns ids and the admin header."""
    token = c.post("/api/admin/setup", json={"name": "Sam", "password": "hunter22"}).json()["token"]
    admin = {"Authorization": f"Bearer {token}"}
    assert c.post("/api/devices/demo", headers=admin).status_code == 201
    kid = c.post("/api/people", json={"name": "Kid", "role": "child"}, headers=admin).json()
    people = {p["name"]: p["id"] for p in c.get("/api/people").json()}
    return {"admin": admin, "sam": people["Sam"], "kid": kid["id"]}


def _see(*people: tuple[int, str], room: str = "Entry", name: str = "front_door"):
    cam = FakeCamera(name, room, [Match(pid, n, 0.1) for pid, n in people])
    cameras._pipelines[name] = cam
    return cam


# ------------------------------------------------------------- admin login


def test_setup_then_endpoints_need_admin(client):
    ids = _setup_house(client)
    assert client.get("/api/admin/status").json()["setup_required"] is False
    # Setup can't be repeated, and admin endpoints now require the token.
    assert client.post("/api/admin/setup", json={"name": "X", "password": "secret1"}).status_code == 409
    assert client.post("/api/devices/demo").status_code == 401
    assert client.get("/api/admin/people", headers=ids["admin"]).status_code == 200


def test_login_requires_face_when_enrolled(client):
    ids = _setup_house(client)
    cameras.recognizer = FakeRecognizer()
    with session_scope() as s:
        s.add(FaceEmbedding(person_id=ids["sam"], vector=b"\0" * 8, dim=2, model_name="stub/haar"))

    _see(room="Entry")                                    # camera running, Sam not in view
    r = client.post("/api/admin/login", json={"name": "Sam", "password": "hunter22"})
    assert r.status_code == 403

    _see((ids["sam"], "Sam"))                             # now the camera sees Sam
    r = client.post("/api/admin/login", json={"name": "Sam", "password": "hunter22"})
    assert r.status_code == 200 and r.json()["warnings"] == []

    r = client.post("/api/admin/login", json={"name": "Sam", "password": "wrong"})
    assert r.status_code == 401


def test_password_hash_roundtrip():
    stored = auth.hash_password("correct horse")
    assert auth.check_password("correct horse", stored)
    assert not auth.check_password("wrong", stored)
    assert "correct" not in stored


def test_cant_demote_only_admin(client):
    ids = _setup_house(client)
    r = client.patch(f"/api/admin/people/{ids['sam']}", json={"role": "adult"}, headers=ids["admin"])
    assert r.status_code == 409


# ------------------------------------------------- camera-checked control


def test_child_can_use_lights_but_not_garage(client):
    ids = _setup_house(client)
    _see((ids["kid"], "Kid"), room="Garage", name="garage_cam")
    _see((ids["kid"], "Kid"), room="Entry", name="front_door")

    r = client.post("/api/devices/light.entry/command", json={"payload": {"state": "ON"}})
    assert r.status_code == 200 and r.json()["by"] == "Kid"

    r = client.post("/api/devices/garage.main/command", json={"payload": {"state": "OPEN"}})
    assert r.status_code == 403
    assert "Kid" in r.json()["detail"]
    denied = client.get("/api/events?type=access_denied").json()
    assert denied and denied[0]["device_id"] == "garage.main"


def test_nobody_at_camera_is_refused(client):
    _setup_house(client)
    _see(room="Entry")
    r = client.post("/api/devices/light.entry/command", json={"payload": {"state": "ON"}})
    assert r.status_code == 403


def test_closest_camera_is_the_one_in_the_devices_room(client):
    ids = _setup_house(client)
    # The kid is visible only in the living room; the garage camera sees nobody.
    _see(room="Garage", name="garage_cam")
    _see((ids["kid"], "Kid"), room="Living Room", name="living_cam")
    assert client.post("/api/devices/light.living_room/command",
                       json={"payload": {"state": "ON"}}).status_code == 200
    r = client.post("/api/devices/garage.main/command", json={"payload": {"state": "OPEN"}})
    assert r.status_code == 403 and "Nobody recognized" in r.json()["detail"]


def test_admin_override_beats_role_default(client):
    ids = _setup_house(client)
    _see((ids["kid"], "Kid"), room="Living Room", name="living_cam")
    h = ids["admin"]
    url = f"/api/admin/people/{ids['kid']}/access"

    # Allow the AC for the kid, then block just the TV.
    client.put(url, json={"target": "kind:thermostat", "allowed": True}, headers=h)
    client.put(url, json={"target": "device:switch.tv", "allowed": False}, headers=h)
    cmd = lambda d, p: client.post(f"/api/devices/{d}/command", json={"payload": p}).status_code
    assert cmd("thermostat.main", {"mode": "cool", "target": 70}) == 200
    assert cmd("switch.tv", {"state": "ON"}) == 403

    # Reset the TV rule: back to the child default (switches allowed).
    client.put(url, json={"target": "device:switch.tv", "allowed": None}, headers=h)
    assert cmd("switch.tv", {"state": "ON"}) == 200


def test_admin_bypasses_camera(client):
    ids = _setup_house(client)
    r = client.post("/api/devices/garage.main/command",
                    json={"payload": {"state": "OPEN"}}, headers=ids["admin"])
    assert r.status_code == 200 and r.json()["via"] == "admin"


# ------------------------------------------------------- device registry


def test_add_switch_transport_and_remove_device(client):
    ids = _setup_house(client)
    h = ids["admin"]
    dev = client.post("/api/devices", json={"name": "Bedroom Lamp", "kind": "light", "room": "Bedroom"},
                      headers=h).json()
    assert dev["id"] == "light.bedroom_lamp" and dev["transport"] == "simulated"

    r = client.patch(f"/api/devices/{dev['id']}", json={"transport": "mqtt"}, headers=h)
    assert r.json()["transport"] == "mqtt"
    assert client.delete(f"/api/devices/{dev['id']}", headers=h).status_code == 204
    assert all(d["id"] != dev["id"] for d in client.get("/api/devices").json())


def test_bad_command_is_rejected(client):
    ids = _setup_house(client)
    r = client.post("/api/devices/thermostat.main/command",
                    json={"payload": {"target": 120}}, headers=ids["admin"])
    assert r.status_code == 400


# ------------------------------------------------------ arrival preferences


def test_arrival_applies_only_permitted_preferences(client):
    ids = _setup_house(client)
    h = ids["admin"]
    base = f"/api/admin/people/{ids['kid']}/preferences"
    client.put(f"{base}/light.living_room", json={"payload": {"brightness": 40}}, headers=h)
    r = client.put(f"{base}/thermostat.main", json={"payload": {"mode": "cool", "target": 68}}, headers=h)
    assert r.json()["warning"]                       # child can't use the AC

    from app.events import Event, bus

    asyncio.run(bus.publish(Event(type=EventType.PRESENCE_ARRIVED, source="vision:front_door",
                                  person_id=ids["kid"], payload={"name": "Kid"})))
    state = {d["id"]: d["state"] for d in client.get("/api/devices").json()}
    assert state["light.living_room"]["brightness"] == 40
    assert state["light.living_room"]["state"] == "ON"
    assert state["thermostat.main"]["mode"] == "off"   # untouched


# ------------------------------------------------------------ simulator


def test_light_dimming_rules():
    assert devices.normalize("light", {"brightness": 30}) == {"brightness": 30, "state": "ON"}
    assert devices.normalize("light", {"brightness": 0}) == {"brightness": 0, "state": "OFF"}
    assert devices.normalize("light", {"state": "TOGGLE"}, {"state": "ON"}) == {"state": "OFF"}
    with pytest.raises(devices.CommandError):
        devices.normalize("motion_sensor", {"state": "ON"})


def test_ac_moves_room_toward_target():
    state = {"mode": "cool", "target": 70, "current": 71}
    while _step_thermostat(state):
        pass
    assert state["current"] == 70
    off = {"mode": "off", "target": 70, "current": 77.9}
    while _step_thermostat(off):
        pass
    assert off["current"] == 78.0                    # drifts to ambient and stops


def test_garage_opens_through_opening(monkeypatch):
    monkeypatch.setattr("app.iot.simulator.GARAGE_TRAVEL_SECONDS", 0.01)
    seen: list = []

    async def run():
        sim = Simulator()
        monkeypatch.setattr("app.iot.simulator.bus.publish",
                            lambda e: seen.append(e.payload["state"]) or asyncio.sleep(0))
        await sim.add("garage.x", "garage")
        await sim.apply("garage.x", {"state": "OPEN"})
        await asyncio.sleep(0.05)

    asyncio.run(run())
    assert seen == ["CLOSED", "OPENING", "OPEN"]


# ----------------------------------------------------------- face precision


def test_gallery_rejects_ambiguous_match():
    gallery = FaceGallery(threshold=0.5, margin=0.1)
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.8, 0.6], dtype=np.float32)       # 0.2 away from A
    gallery.load([(1, "A", a), (2, "B", b)])
    probe = (a + b) / np.linalg.norm(a + b)          # halfway: too close to call
    assert gallery.match(probe).person_id is None
    assert gallery.match(a).person_id == 1


def test_identity_needs_agreeing_frames(monkeypatch):
    from app.vision import pipeline as pl
    from app.vision.detector import Detection

    sent: list = []
    monkeypatch.setattr("app.vision.pipeline.bus.publish_threadsafe", sent.append)
    vp = pl.VisionPipeline()
    track = pl.TrackState()
    det = Detection("person", 0.9, (0, 0, 10, 10), track_id=1)
    alice = Match(1, "Alice", 0.2)

    vp._vote(track, alice, det, "vision:test")
    vp._vote(track, Match(None, "unknown", 0.7), det, "vision:test")   # one bad frame
    vp._vote(track, alice, det, "vision:test")
    assert track.identity is None and sent == []                       # 2 of 3: not yet
    vp._vote(track, alice, det, "vision:test")
    assert track.identity.person_id == 1
    assert [e.type for e in sent] == [EventType.PERSON_RECOGNIZED]


def test_old_database_gets_new_columns(tmp_path):
    """A database from before roles existed is upgraded in place."""
    import sqlite3

    from sqlalchemy import create_engine

    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, name VARCHAR(80), is_household BOOLEAN,"
                " is_home BOOLEAN, last_seen DATETIME, created_at DATETIME)")
    con.execute("INSERT INTO people (name, is_household, is_home) VALUES ('Old', 1, 0)")
    con.commit()
    con.close()

    import app.db.database as database

    old_engine = create_engine(f"sqlite:///{path.as_posix()}")
    original = database.engine
    database.engine = old_engine
    try:
        database.init_db()
    finally:
        database.engine = original
    con = sqlite3.connect(path)
    assert con.execute("SELECT role, password_hash FROM people").fetchone() == ("adult", None)
    con.close()


def test_mqtt_state_only_counts_once_device_is_real(client):
    ids = _setup_house(client)
    from app.events import Event, bus

    def mqtt_report():
        asyncio.run(bus.publish(Event(type=EventType.DEVICE_STATE, source="mqtt:homeai/device/switch.tv/state",
                                      device_id="switch.tv", payload={"kind": "switch", "state": "ON"})))
        return {d["id"]: d for d in client.get("/api/devices").json()}["switch.tv"]

    assert mqtt_report()["state"]["state"] == "OFF"          # still simulated: ignored
    client.patch("/api/devices/switch.tv", json={"transport": "mqtt"}, headers=ids["admin"])
    tv = mqtt_report()
    assert tv["state"] == {"state": "ON"} and tv["online"] is True


# ------------------------------------------------------------ door arrivals


def test_walking_in_is_the_arrival_not_standing_in_view(client, monkeypatch):
    from app.config import settings
    from app.events import Event, bus

    monkeypatch.setattr(settings, "door_lines", "front_door:0.1,0.6,0.9,0.6")
    ids = _setup_house(client)
    arrivals: list = []

    async def record(e):
        arrivals.append(e)

    bus.subscribe(record, {EventType.PRESENCE_ARRIVED})
    home = lambda: {p["name"]: p["is_home"] for p in client.get("/api/people").json()}

    # Recognized by the door camera while standing there: not an arrival.
    asyncio.run(bus.publish(Event(type=EventType.PERSON_RECOGNIZED, source="vision:front_door",
                                  person_id=ids["kid"], payload={"name": "Kid", "camera": "front_door"})))
    assert home()["Kid"] is False and arrivals == []

    # Walks in across the line: one arrival, with the name.
    entered = Event(type=EventType.PERSON_ENTERED, source="vision:front_door", person_id=ids["kid"],
                    payload={"name": "Kid", "face": "known", "camera": "front_door"})
    asyncio.run(bus.publish(entered))
    asyncio.run(bus.publish(entered))                    # a second crossing while home
    assert home()["Kid"] is True
    assert [(e.payload["name"], e.payload["via"]) for e in arrivals] == [("Kid", "door")]

    # Other cameras still count a sighting as being home.
    asyncio.run(bus.publish(Event(type=EventType.PERSON_RECOGNIZED, source="vision:living_room",
                                  person_id=ids["sam"], payload={"name": "Sam", "camera": "living_room"})))
    assert home()["Sam"] is True


def test_unrecognized_entry_raises_an_alert(client, monkeypatch):
    from app.events import Event, bus
    from app.notifications import notifier

    sent: list = []
    monkeypatch.setattr(notifier, "send", lambda msg, priority="normal", **kw: sent.append((msg, priority)))
    asyncio.run(bus.publish(Event(type=EventType.PERSON_ENTERED, source="vision:front_door",
                                  payload={"name": None, "face": "none", "camera": "front_door"})))
    assert sent == [("Someone came in at front_door (no face seen).", "high")]


# ------------------------------------------------------- door line editor


def test_admin_draws_a_door_line_that_takes_effect_and_persists(client):
    from app.vision.doorway import door_lines

    ids = _setup_house(client)
    cam = _see(room="Entry", name="front_door")
    url = "/api/cameras/front_door/door"
    line = {"x1": 0.1, "y1": 0.7, "x2": 0.9, "y2": 0.7}

    assert client.put(url, json=line).status_code == 401              # admin only
    assert client.put(url, json={**line, "x2": 0.12, "y2": 0.71}, headers=ids["admin"]).status_code == 400
    assert client.put(url, json={**line, "x1": 1.5}, headers=ids["admin"]).status_code == 422
    assert client.put("/api/cameras/nope/door", json=line, headers=ids["admin"]).status_code == 404

    r = client.put(url, json={**line, "flip": True}, headers=ids["admin"])
    assert r.status_code == 200
    assert cam.door is not None and cam.door.flip                   # live, no restart
    assert door_lines()["front_door"].y1 == 0.7                     # saved
    [listed] = client.get("/api/cameras").json()
    assert listed["door"] == {**line, "flip": True}


def test_removing_a_line_overrides_the_env_setting(client, monkeypatch):
    from app.config import settings
    from app.vision.doorway import door_lines

    monkeypatch.setattr(settings, "door_lines", "front_door:0.1,0.6,0.9,0.6")
    ids = _setup_house(client)
    cam = _see(room="Entry", name="front_door")
    assert "front_door" in door_lines()
    assert client.delete("/api/cameras/front_door/door", headers=ids["admin"]).status_code == 204
    assert "front_door" not in door_lines() and cam.door is None



# ------------------------------------------------------------ door exits


def _exit_event(person_id=None, name=None):
    from app.events import Event

    return Event(type=EventType.PERSON_EXITED, source="vision:front_door", person_id=person_id,
                 payload={"name": name, "face": "known" if person_id else "none", "camera": "front_door"})


def _set_home(*person_ids):
    with session_scope() as session:
        for p in session.query(Person).all():
            p.is_home = p.id in person_ids


def test_walking_out_marks_you_away_and_shuts_down_an_empty_house(client):
    from app.events import bus

    ids = _setup_house(client)
    h = ids["admin"]
    client.post("/api/devices/light.living_room/command", json={"payload": {"state": "ON"}}, headers=h)
    _set_home(ids["sam"])
    left: list = []

    async def record(e):
        left.append(e)

    bus.subscribe(record, {EventType.PRESENCE_LEFT})
    asyncio.run(bus.publish(_exit_event(ids["sam"], "Sam")))

    home = {p["name"]: p["is_home"] for p in client.get("/api/people").json()}
    assert home["Sam"] is False
    assert [(e.payload["name"], e.payload["via"], e.payload["guessed"]) for e in left] == [("Sam", "door", False)]
    state = {d["id"]: d["state"] for d in client.get("/api/devices").json()}
    assert state["light.living_room"]["state"] == "OFF"            # nobody home: away_shutdown
    asyncio.run(bus.publish(_exit_event(ids["sam"], "Sam")))       # already away: nothing more
    assert len(left) == 1


def test_faceless_exit_is_the_only_person_home_unless_seen_elsewhere(client):
    from app.events import bus

    ids = _setup_house(client)
    home = lambda: {p["name"]: p["is_home"] for p in client.get("/api/people").json()}

    _set_home(ids["sam"], ids["kid"])                              # two home: can't tell who
    asyncio.run(bus.publish(_exit_event()))
    assert home() == {"Sam": True, "Kid": True}

    _set_home(ids["kid"])
    _see((ids["kid"], "Kid"), room="Living Room", name="living_room")   # ...but still on camera
    asyncio.run(bus.publish(_exit_event()))
    assert home()["Kid"] is True

    cameras._pipelines = {}                                        # out of sight: must be them
    asyncio.run(bus.publish(_exit_event()))
    assert home()["Kid"] is False


def test_presence_timeout_is_a_long_backup_once_there_is_a_door_line(client, monkeypatch):
    from datetime import timedelta

    from app.automation.rules import presence_timeout
    from app.config import settings

    assert presence_timeout() == timedelta(minutes=5)
    monkeypatch.setattr(settings, "door_lines", "front_door:0.1,0.6,0.9,0.6")
    assert presence_timeout() == timedelta(minutes=120)
