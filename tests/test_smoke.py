"""Smoke tests: the backbone holds together without hardware or a broker."""

from __future__ import annotations

import numpy as np

from app.events import Event, EventBus, EventType
from app.vision.recognizer import FaceCropper, FaceGallery, StubEncoder


def test_event_serializes():
    event = Event(type=EventType.PERSON_RECOGNIZED, source="vision:test", payload={"name": "A"})
    data = event.to_dict()
    assert data["type"] == "person_recognized"
    assert "timestamp" in data


def test_bus_filters_by_type():
    import asyncio

    seen: list[Event] = []

    async def handler(event: Event) -> None:
        seen.append(event)

    bus = EventBus()
    bus.subscribe(handler, {EventType.UNKNOWN_PERSON})

    async def run() -> None:
        await bus.publish(Event(type=EventType.SENSOR_READING, source="t"))
        await bus.publish(Event(type=EventType.UNKNOWN_PERSON, source="t"))

    asyncio.run(run())
    assert [e.type for e in seen] == [EventType.UNKNOWN_PERSON]


def test_gallery_matches_known_and_rejects_far():
    gallery = FaceGallery(threshold=0.3)
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    gallery.load([(1, "Alice", a), (2, "Bob", b)])

    assert gallery.match(a).person_id == 1
    assert gallery.match(np.array([0.0, 0.0, 1.0], dtype=np.float32)).person_id is None


def test_empty_gallery_returns_unknown():
    gallery = FaceGallery(threshold=0.6)
    gallery.load([])
    assert gallery.match(np.array([1.0, 0.0], dtype=np.float32)).name == "unknown"


def test_stub_encoder_is_deterministic_and_normalized():
    encoder = StubEncoder()
    image = (np.random.default_rng(0).random((60, 60, 3)) * 255).astype(np.uint8)

    first, second = encoder.embed(image), encoder.embed(image)
    assert np.allclose(first, second)
    assert np.isclose(np.linalg.norm(first), 1.0, atol=1e-5)


def test_cropper_finds_no_face_in_blank_image():
    cropper = FaceCropper()
    assert cropper.crop(np.zeros((200, 200, 3), dtype=np.uint8)) is None
    assert cropper.crop(np.zeros((0, 0, 3), dtype=np.uint8)) is None


def test_recognized_track_sends_rate_limited_heartbeat(monkeypatch):
    from app.vision import pipeline as pl
    from app.vision.detector import Detection
    from app.vision.recognizer import Match

    sent: list = []
    clock = [1000.0]
    monkeypatch.setattr("app.vision.pipeline.bus.publish_threadsafe", sent.append)
    monkeypatch.setattr("app.vision.pipeline.time.monotonic", lambda: clock[0])

    vp = pl.VisionPipeline()
    vp._announced[7] = Match(person_id=1, name="Alice", distance=0.1)
    det = Detection(label="person", confidence=0.9, box=(0, 0, 10, 10), track_id=7)

    vp._heartbeat(det, "vision:test")                 # first sighting -> sent
    clock[0] += 5
    vp._heartbeat(det, "vision:test")                 # too soon -> suppressed
    clock[0] += pl.SEEN_HEARTBEAT_SECONDS
    vp._heartbeat(det, "vision:test")                 # interval passed -> sent

    assert [e.type for e in sent] == [EventType.PERSON_SEEN, EventType.PERSON_SEEN]
    assert sent[0].person_id == 1


def test_unknown_or_untracked_gets_no_heartbeat(monkeypatch):
    from app.vision import pipeline as pl
    from app.vision.detector import Detection
    from app.vision.recognizer import Match

    sent: list = []
    monkeypatch.setattr("app.vision.pipeline.bus.publish_threadsafe", sent.append)

    vp = pl.VisionPipeline()
    vp._announced[3] = Match(person_id=None, name="unknown", distance=0.9)
    vp._heartbeat(Detection("person", 0.9, (0, 0, 10, 10), track_id=3), "vision:test")
    vp._heartbeat(Detection("person", 0.9, (0, 0, 10, 10), track_id=None), "vision:test")
    assert sent == []


# --- MQTT topic parsing -------------------------------------------------------
# Exercised without a broker: feed the bridge a message and capture what it
# would have put on the bus.


class _FakeMessage:
    def __init__(self, topic: str, payload: str) -> None:
        self.topic = topic
        self.payload = payload.encode()


def _parse(monkeypatch, topic: str, payload: str):
    from app.iot.mqtt_client import MqttBridge

    captured: list = []
    monkeypatch.setattr("app.iot.mqtt_client.bus.publish_threadsafe", captured.append)
    MqttBridge()._on_message(None, None, _FakeMessage(topic, payload))
    return captured


def test_device_state_topic_becomes_event(monkeypatch):
    events = _parse(monkeypatch, "homeai/device/light.entry/state", '{"state": "ON"}')
    assert len(events) == 1
    assert events[0].type == EventType.DEVICE_STATE
    assert events[0].device_id == "light.entry"
    assert events[0].payload == {"state": "ON"}


def test_sensor_topic_becomes_event(monkeypatch):
    events = _parse(monkeypatch, "homeai/sensor/sensor.kitchen", '{"temperature": 21.5}')
    assert len(events) == 1
    assert events[0].type == EventType.SENSOR_READING
    assert events[0].device_id == "sensor.kitchen"


def test_malformed_payload_does_not_crash(monkeypatch):
    events = _parse(monkeypatch, "homeai/sensor/sensor.kitchen", "not json")
    assert events[0].payload == {"raw": "not json"}


def test_unknown_topic_is_ignored(monkeypatch):
    assert _parse(monkeypatch, "homeai/something/else", "{}") == []
