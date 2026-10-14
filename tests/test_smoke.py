"""Smoke tests: the backbone holds together without hardware or a broker."""

from __future__ import annotations

import numpy as np

from app.events import Event, EventBus, EventType
from app.vision.recognizer import FaceGallery, StubEncoder


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
