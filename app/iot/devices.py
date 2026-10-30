"""Device kinds, command validation, and command routing.

Everything that wants to change a device -- the dashboard, the rules engine, an
arrival preference -- calls `send()`. It validates the command for the device's
kind, then delivers it to the simulator or to real hardware over MQTT depending
on the device's `transport`. Callers never know which one they got, which is what
lets a simulated device be swapped for a real one without touching anything else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.db import session_scope
from app.db.models import Device
from app.events import Event, EventType, bus

log = logging.getLogger(__name__)


class CommandError(ValueError):
    """The command doesn't make sense for this device."""


@dataclass(frozen=True)
class Kind:
    label: str
    controllable: bool                  # sensors report; they can't be commanded
    default_state: dict[str, Any] = field(default_factory=dict)


KINDS: dict[str, Kind] = {
    "light": Kind("Light", True, {"state": "OFF", "brightness": 100}),
    "switch": Kind("Switch / plug", True, {"state": "OFF"}),
    "thermostat": Kind("AC / thermostat", True, {"mode": "off", "target": 72, "current": 76}),
    "lock": Kind("Door lock", True, {"state": "LOCKED"}),
    "garage": Kind("Garage door", True, {"state": "CLOSED"}),
    "motion_sensor": Kind("Motion sensor", False, {"motion": False}),
    "contact_sensor": Kind("Door / window sensor", False, {"contact": "closed"}),
    "temperature_sensor": Kind("Temperature sensor", False, {"temperature": 72.0, "humidity": 40}),
}

THERMOSTAT_MODES = ("off", "cool", "heat", "auto")
THERMOSTAT_RANGE = (60, 85)    # degrees F


def normalize(kind: str, payload: dict[str, Any], current: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate a command and return the clean version. Raises CommandError."""
    spec = KINDS.get(kind)
    if spec is None:
        raise CommandError(f"unknown device kind '{kind}'")
    if not spec.controllable:
        raise CommandError(f"a {spec.label.lower()} can't be controlled")
    current = current or {}
    out: dict[str, Any] = {}

    def on_off(value: Any) -> str:
        value = str(value).upper()
        if value == "TOGGLE":
            return "OFF" if current.get("state") == "ON" else "ON"
        if value not in ("ON", "OFF"):
            raise CommandError("state must be ON, OFF or TOGGLE")
        return value

    if kind in ("light", "switch"):
        if "state" in payload:
            out["state"] = on_off(payload["state"])
        if kind == "light" and "brightness" in payload:
            try:
                level = int(payload["brightness"])
            except (TypeError, ValueError):
                raise CommandError("brightness must be a number from 0 to 100") from None
            if not 0 <= level <= 100:
                raise CommandError("brightness must be from 0 to 100")
            out["brightness"] = level
            # Dimming to zero is off; dimming up from off turns the light on.
            if "state" not in out:
                out["state"] = "ON" if level > 0 else "OFF"

    elif kind == "thermostat":
        if "mode" in payload:
            mode = str(payload["mode"]).lower()
            if mode not in THERMOSTAT_MODES:
                raise CommandError(f"mode must be one of {', '.join(THERMOSTAT_MODES)}")
            out["mode"] = mode
        if "target" in payload:
            try:
                target = int(round(float(payload["target"])))
            except (TypeError, ValueError):
                raise CommandError("target must be a temperature") from None
            low, high = THERMOSTAT_RANGE
            if not low <= target <= high:
                raise CommandError(f"target must be between {low} and {high} F")
            out["target"] = target

    elif kind == "lock":
        state = str(payload.get("state", "")).upper()
        if state not in ("LOCKED", "UNLOCKED"):
            raise CommandError("state must be LOCKED or UNLOCKED")
        out["state"] = state

    elif kind == "garage":
        state = str(payload.get("state", "")).upper()
        if state not in ("OPEN", "CLOSED"):
            raise CommandError("state must be OPEN or CLOSED")
        out["state"] = state

    if not out:
        raise CommandError("command changes nothing")
    return out


async def send(device_id: str, payload: dict[str, Any], source: str, actor: str | None = None) -> dict[str, Any]:
    """Validate and deliver a command. Raises CommandError / LookupError."""
    from app.iot.mqtt_client import mqtt_bridge
    from app.iot.simulator import simulator

    with session_scope() as session:
        device = session.get(Device, device_id)
        if device is None:
            raise LookupError(f"device '{device_id}' is not registered")
        kind, transport, state = device.kind, device.transport, dict(device.state or {})

    command = normalize(kind, payload, state)
    if transport == "simulated":
        await simulator.apply(device_id, command)
        delivered = True
    else:
        delivered = mqtt_bridge.command(device_id, command)

    await bus.publish(
        Event(
            type=EventType.DEVICE_COMMAND,
            source=source,
            device_id=device_id,
            payload={**command, "delivered": delivered, "transport": transport, "by": actor},
        )
    )
    return {"device_id": device_id, "payload": command, "delivered": delivered, "transport": transport}
