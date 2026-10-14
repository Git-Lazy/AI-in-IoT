"""Simulated IoT devices for development.

Publishes state and sensor readings on the project's MQTT topics and listens for
commands, so the backend, rules engine, and dashboard can be exercised end-to-end
before real hardware arrives.

    python scripts/fake_device.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paho.mqtt.client as mqtt  # noqa: E402

from app.config import settings  # noqa: E402

BASE = settings.mqtt_base_topic.rstrip("/")

DEVICES = {
    "light.entry": {"kind": "light", "state": "OFF"},
    "light.porch": {"kind": "light", "state": "OFF"},
    "lock.front_door": {"kind": "lock", "state": "LOCKED"},
}
SENSORS = {
    "sensor.kitchen": lambda: {"temperature": round(random.uniform(19, 24), 1),
                               "humidity": round(random.uniform(35, 55), 1)},
    "sensor.front_door": lambda: {"contact": random.choice(["open", "closed"])},
}


def on_connect(client, userdata, flags, rc, properties=None):
    print(f"[fake-device] connected to {settings.mqtt_host}:{settings.mqtt_port}")
    client.subscribe(f"{BASE}/device/+/set")
    for device_id, state in DEVICES.items():
        publish_state(client, device_id, state)


def on_message(client, userdata, msg):
    device_id = msg.topic.split("/")[2]
    if device_id not in DEVICES:
        return
    command = json.loads(msg.payload.decode())
    print(f"[fake-device] {device_id} <- {command}")
    DEVICES[device_id].update(command)
    publish_state(client, device_id, DEVICES[device_id])


def publish_state(client, device_id: str, state: dict) -> None:
    client.publish(f"{BASE}/device/{device_id}/state", json.dumps(state), qos=1, retain=True)


def main() -> None:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="fake-devices")
    if settings.mqtt_username:
        client.username_pw_set(settings.mqtt_username, settings.mqtt_password)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(settings.mqtt_host, settings.mqtt_port, 60)
    client.loop_start()

    try:
        while True:
            for sensor_id, reading in SENSORS.items():
                payload = reading()
                client.publish(f"{BASE}/sensor/{sensor_id}", json.dumps(payload), qos=0)
                print(f"[fake-device] {sensor_id} -> {payload}")
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n[fake-device] stopping")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
