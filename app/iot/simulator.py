"""Simulated IoT devices that live inside the backend.

Lets the whole platform -- permissions, arrival preferences, the dashboard -- run
end-to-end before any hardware is bought, and without an MQTT broker. Each
simulated device behaves like the real thing would: lights dim, the AC pulls the
room temperature toward its target over time, the garage door takes a few
seconds to open, motion clears after a while.

A simulated device reports state exactly the way an MQTT device does (as
DEVICE_STATE / SENSOR_READING events), so nothing downstream can tell the
difference. Flip the device's transport to "mqtt" when the real one is wired up.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

from app.db import session_scope
from app.db.models import Device
from app.events import Event, EventType, bus
from app.iot.devices import KINDS

log = logging.getLogger(__name__)

TICK_SECONDS = 5
AMBIENT_F = 78.0               # where the room drifts with the AC off
AC_STEP_F = 0.5                # how fast the AC moves the room per tick
GARAGE_TRAVEL_SECONDS = 4
MOTION_CLEAR_SECONDS = 15
TEMPERATURE_REPORT_SECONDS = 30


class Simulator:
    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}
        self._kinds: dict[str, str] = {}
        self._motion_at: dict[str, float] = {}
        self._last_temp_report = 0.0
        self._task: asyncio.Task | None = None

    # ---------------------------------------------------------------- registry

    def load(self) -> None:
        """Pick up every simulated device from the database."""
        with session_scope() as session:
            for device in session.query(Device).filter(Device.transport == "simulated"):
                self._register(device.id, device.kind, device.state)
                # Fill in defaults so a never-used device still shows a state.
                device.state = dict(self._states[device.id])
                device.online = True
        log.info("simulator: %d device(s)", len(self._states))

    def _register(self, device_id: str, kind: str, state: dict | None) -> None:
        spec = KINDS.get(kind)
        defaults = dict(spec.default_state) if spec else {}
        self._kinds[device_id] = kind
        self._states[device_id] = {**defaults, **(state or {})}

    async def add(self, device_id: str, kind: str, state: dict | None = None) -> None:
        self._register(device_id, kind, state)
        await self._report(device_id)

    def remove(self, device_id: str) -> None:
        self._states.pop(device_id, None)
        self._kinds.pop(device_id, None)
        self._motion_at.pop(device_id, None)

    def has(self, device_id: str) -> bool:
        return device_id in self._states

    # ----------------------------------------------------------------- control

    async def apply(self, device_id: str, command: dict[str, Any]) -> None:
        """Apply an already-validated command (see devices.normalize)."""
        if device_id not in self._states:
            raise LookupError(f"'{device_id}' is not a simulated device")
        state = self._states[device_id]

        if self._kinds[device_id] == "garage":
            target = command["state"]
            if state["state"] != target:
                state["state"] = "OPENING" if target == "OPEN" else "CLOSING"
                asyncio.create_task(self._finish_garage(device_id, target))
        else:
            state.update(command)
        await self._report(device_id)

    async def trigger(self, device_id: str, reading: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fake a sensor event -- someone walks past, a door opens."""
        if device_id not in self._states:
            raise LookupError(f"'{device_id}' is not a simulated device")
        kind = self._kinds[device_id]
        state = self._states[device_id]

        if kind == "motion_sensor":
            state["motion"] = True
            self._motion_at[device_id] = time.monotonic()
        elif kind == "contact_sensor":
            state["contact"] = "open" if state.get("contact") == "closed" else "closed"
        elif kind == "temperature_sensor" and reading and "temperature" in reading:
            state["temperature"] = float(reading["temperature"])
        else:
            raise ValueError(f"nothing to trigger on a {kind}")
        await self._report(device_id)
        return dict(state)

    async def _finish_garage(self, device_id: str, target: str) -> None:
        await asyncio.sleep(GARAGE_TRAVEL_SECONDS)
        state = self._states.get(device_id)
        if state is not None and state["state"] in ("OPENING", "CLOSING"):
            state["state"] = target
            await self._report(device_id)

    # -------------------------------------------------------------------- time

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(TICK_SECONDS)
            try:
                await self.tick()
            except Exception:
                log.exception("simulator tick failed")

    async def tick(self) -> None:
        now = time.monotonic()
        report_temps = now - self._last_temp_report >= TEMPERATURE_REPORT_SECONDS
        if report_temps:
            self._last_temp_report = now

        for device_id, kind in list(self._kinds.items()):
            state = self._states[device_id]
            changed = False

            if kind == "thermostat":
                changed = _step_thermostat(state)
            elif kind == "motion_sensor" and state.get("motion"):
                if now - self._motion_at.get(device_id, 0) >= MOTION_CLEAR_SECONDS:
                    state["motion"] = False
                    changed = True
            elif kind == "temperature_sensor" and report_temps:
                state["temperature"] = round(state["temperature"] + random.uniform(-0.3, 0.3), 1)
                state["humidity"] = max(20, min(70, state["humidity"] + random.randint(-1, 1)))
                changed = True

            if changed:
                await self._report(device_id)

    async def _report(self, device_id: str) -> None:
        kind = self._kinds[device_id]
        sensor = not KINDS[kind].controllable if kind in KINDS else False
        await bus.publish(
            Event(
                type=EventType.SENSOR_READING if sensor else EventType.DEVICE_STATE,
                source=f"sim:{device_id}",
                device_id=device_id,
                payload={"kind": kind, **self._states[device_id]},
            )
        )


def _step_thermostat(state: dict[str, Any]) -> bool:
    """Move the room temperature one tick toward wherever the AC is pushing it."""
    current, target, mode = float(state["current"]), float(state["target"]), state["mode"]
    if (mode == "cool" and current > target) or (mode == "heat" and current < target) \
            or (mode == "auto" and current != target):
        new = _toward(current, target, AC_STEP_F)
    elif mode == "off" and current != AMBIENT_F:
        new = _toward(current, AMBIENT_F, 0.2)
    else:
        return False
    state["current"] = round(new, 1)
    return True


def _toward(value: float, goal: float, step: float) -> float:
    return min(goal, value + step) if value < goal else max(goal, value - step)


simulator = Simulator()
