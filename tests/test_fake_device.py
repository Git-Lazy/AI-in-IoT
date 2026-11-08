"""The MQTT stand-in device (scripts/fake_device.py), without a broker."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parent.parent / "scripts" / "fake_device.py"
_spec = importlib.util.spec_from_file_location("fake_device", _path)
fake_device = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fake_device)


@pytest.fixture
def house():
    sent: list[tuple[str, dict]] = []
    devices = fake_device.FakeDevices(
        {
            "light.porch": ("light", {"brightness": 50}),
            "thermostat.main": ("thermostat", {"current": 75}),
            "garage.main": ("garage", None),
            "sensor.hall": ("motion_sensor", None),
        },
        lambda topic, payload: sent.append((topic, payload)),
    )
    return devices, sent


def test_starts_from_last_known_state_over_defaults(house):
    devices, _ = house
    assert devices.states["light.porch"] == {"state": "OFF", "brightness": 50}
    assert devices.states["thermostat.main"]["current"] == 75


def test_command_changes_state_and_reports_back(house):
    devices, sent = house
    devices.on_command("light.porch", {"state": "ON", "brightness": 80})
    topic, payload = sent[-1]
    assert topic == "homeai/device/light.porch/state"
    assert payload == {"kind": "light", "state": "ON", "brightness": 80}


def test_sensors_report_on_sensor_topic_and_ignore_commands(house):
    devices, sent = house
    devices.on_command("sensor.hall", {"motion": True})
    assert sent == []
    devices.press("sensor.hall")
    assert sent[-1] == ("homeai/sensor/sensor.hall", {"kind": "motion_sensor", "motion": True})


def test_unknown_device_commands_are_ignored(house):
    devices, sent = house
    devices.on_command("light.someone_elses", {"state": "ON"})
    assert sent == []


def test_garage_travels(house, monkeypatch):
    monkeypatch.setattr(fake_device, "GARAGE_TRAVEL_SECONDS", 0.01)
    devices, sent = house
    devices.on_command("garage.main", {"state": "OPEN"})
    time.sleep(0.1)
    assert [p["state"] for t, p in sent if "garage" in t] == ["OPENING", "OPEN"]


def test_ac_and_motion_advance_with_time(house, monkeypatch):
    devices, _ = house
    devices.on_command("thermostat.main", {"mode": "cool", "target": 70})
    devices.tick()
    assert devices.states["thermostat.main"]["current"] == 74.5

    monkeypatch.setattr(fake_device, "MOTION_CLEAR_SECONDS", 0)
    devices.press("sensor.hall")
    devices.tick()
    assert devices.states["sensor.hall"]["motion"] is False
