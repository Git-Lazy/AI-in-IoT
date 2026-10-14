"""MQTT bridge between IoT devices and the event bus.

Topic convention (base topic defaults to "homeai"):
    homeai/device/<device_id>/state     device -> backend, JSON state
    homeai/device/<device_id>/set       backend -> device, JSON command
    homeai/sensor/<sensor_id>           device -> backend, JSON reading
    homeai/event                        backend -> anyone (Home Assistant, Frigate-style)

Inbound messages become Events; outbound Events become published messages.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import paho.mqtt.client as mqtt

from app.config import settings
from app.events import Event, EventType, bus

log = logging.getLogger(__name__)


class MqttBridge:
    def __init__(self) -> None:
        self.base = settings.mqtt_base_topic.rstrip("/")
        self.connected = False
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=settings.mqtt_client_id,
        )
        if settings.mqtt_username:
            self._client.username_pw_set(settings.mqtt_username, settings.mqtt_password)

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        try:
            self._client.connect(settings.mqtt_host, settings.mqtt_port, keepalive=60)
        except OSError as exc:
            # The platform must still boot without a broker -- useful in Week 2
            # before Mosquitto is running, and it keeps the dashboard testable.
            log.warning("MQTT broker unreachable (%s); running without IoT messaging", exc)
            return
        self._client.loop_start()

    def stop(self) -> None:
        if self.connected:
            self._client.loop_stop()
            self._client.disconnect()

    # ----------------------------------------------------------------- inbound

    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        if reason_code != 0:
            log.error("MQTT connection refused: %s", reason_code)
            return
        self.connected = True
        log.info("MQTT connected to %s:%s", settings.mqtt_host, settings.mqtt_port)
        client.subscribe([(f"{self.base}/device/+/state", 0), (f"{self.base}/sensor/+", 0)])

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None) -> None:
        self.connected = False
        log.warning("MQTT disconnected (%s)", reason_code)

    def _on_message(self, client, userdata, msg: mqtt.MQTTMessage) -> None:
        try:
            payload: dict[str, Any] = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {"raw": msg.payload.decode(errors="replace")}

        parts = msg.topic.split("/")
        if len(parts) >= 4 and parts[1] == "device":
            event = Event(
                type=EventType.DEVICE_STATE,
                source=f"mqtt:{msg.topic}",
                device_id=parts[2],
                payload=payload,
            )
        elif len(parts) >= 3 and parts[1] == "sensor":
            event = Event(
                type=EventType.SENSOR_READING,
                source=f"mqtt:{msg.topic}",
                device_id=parts[2],
                payload=payload,
            )
        else:
            return

        bus.publish_threadsafe(event)

    # ---------------------------------------------------------------- outbound

    def command(self, device_id: str, payload: dict[str, Any]) -> bool:
        """Send a command to a device. Returns False if the broker is absent."""
        if not self.connected:
            log.warning("cannot command %s: MQTT not connected", device_id)
            return False
        self._client.publish(f"{self.base}/device/{device_id}/set", json.dumps(payload), qos=1)
        log.info("-> %s %s", device_id, payload)
        return True

    def announce(self, event: Event) -> None:
        """Republish an internal event so external systems (Home Assistant) can see it."""
        if not self.connected:
            return
        self._client.publish(f"{self.base}/event", json.dumps(event.to_dict()), qos=0)


mqtt_bridge = MqttBridge()
