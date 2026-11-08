"""A stand-in for real smart-home hardware, talking to the backend over MQTT only.

The backend's built-in simulator never touches MQTT. This script is the other
half: a separate program that behaves like a Shelly plug, a smart bulb or a
thermostat would -- it subscribes to its command topic, changes its state, and
reports back. The backend can't tell it from real hardware, so it proves the
whole chain: dashboard -> backend -> Mosquitto -> device -> Mosquitto -> dashboard.

    1. Start Mosquitto            docker compose up -d mosquitto
    2. On the dashboard (Admin -> Devices), switch a device to "real" (MQTT)
    3. Run                        python scripts/fake_device.py
       or pick devices            python scripts/fake_device.py light.living_room thermostat.main
       or without the window      python scripts/fake_device.py --no-gui

With no device ids, every device set to "real" in the database is emulated. A
window shows each one -- the bulb glows with its brightness -- and lets you act
as the physical world: walk past a motion sensor, open a door, flip a switch.

Topics (see app/iot/mqtt_client.py):
    homeai/device/<id>/set      backend -> device   JSON command
    homeai/device/<id>/state    device  -> backend  JSON full state (retained)
    homeai/sensor/<id>          sensor  -> backend  JSON reading
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paho.mqtt.client as mqtt  # noqa: E402

from app.config import settings  # noqa: E402
from app.iot.devices import KINDS  # noqa: E402
from app.iot.simulator import (  # noqa: E402
    GARAGE_TRAVEL_SECONDS,
    MOTION_CLEAR_SECONDS,
    TICK_SECONDS,
    _step_thermostat,
)

BASE = settings.mqtt_base_topic.rstrip("/")


class FakeDevices:
    """The devices' state and behavior. `publish(topic, payload)` sends a message."""

    def __init__(self, devices: dict[str, tuple[str, dict]], publish) -> None:
        self.lock = threading.Lock()
        self.kinds = {device_id: kind for device_id, (kind, _) in devices.items()}
        self.states = {
            device_id: {**KINDS[kind].default_state, **(state or {})}
            for device_id, (kind, state) in devices.items()
        }
        self.log: list[str] = []
        self._publish = publish
        self._motion_at: dict[str, float] = {}

    # ------------------------------------------------------------- reporting

    def is_sensor(self, device_id: str) -> bool:
        return not KINDS[self.kinds[device_id]].controllable

    def report(self, device_id: str) -> None:
        with self.lock:
            payload = {"kind": self.kinds[device_id], **self.states[device_id]}
        topic = (f"{BASE}/sensor/{device_id}" if self.is_sensor(device_id)
                 else f"{BASE}/device/{device_id}/state")
        self._publish(topic, payload)

    def report_all(self) -> None:
        for device_id in self.states:
            self.report(device_id)

    def note(self, line: str) -> None:
        line = f"{time.strftime('%H:%M:%S')}  {line}"
        print(f"[fake-device] {line}")
        with self.lock:
            self.log.append(line)
            del self.log[:-50]

    # --------------------------------------------------------------- inbound

    def on_command(self, device_id: str, command: dict[str, Any]) -> None:
        """A command from the backend. It already validated it (devices.normalize)."""
        if device_id not in self.states or self.is_sensor(device_id):
            return
        self.note(f"{device_id} <- {command}")
        kind = self.kinds[device_id]
        with self.lock:
            state = self.states[device_id]
            if kind == "garage":
                target = command.get("state")
                if target in ("OPEN", "CLOSED") and state["state"] != target:
                    state["state"] = "OPENING" if target == "OPEN" else "CLOSING"
                    threading.Timer(GARAGE_TRAVEL_SECONDS, self._finish_garage,
                                    args=(device_id, target)).start()
            else:
                state.update(command)
        self.report(device_id)

    def _finish_garage(self, device_id: str, target: str) -> None:
        with self.lock:
            state = self.states[device_id]
            if state["state"] not in ("OPENING", "CLOSING"):
                return
            state["state"] = target
        self.note(f"{device_id} is {target.lower()}")
        self.report(device_id)

    # --------------------------------------------- the physical world (you)

    def press(self, device_id: str) -> None:
        """Someone uses the device by hand -- the wall switch, not the app."""
        kind = self.kinds[device_id]
        with self.lock:
            state = self.states[device_id]
            if kind in ("light", "switch"):
                state["state"] = "OFF" if state["state"] == "ON" else "ON"
            elif kind == "lock":
                state["state"] = "UNLOCKED" if state["state"] == "LOCKED" else "LOCKED"
            elif kind == "motion_sensor":
                state["motion"] = True
                self._motion_at[device_id] = time.monotonic()
            elif kind == "contact_sensor":
                state["contact"] = "open" if state["contact"] == "closed" else "closed"
            elif kind == "temperature_sensor":
                state["temperature"] = round(state["temperature"] + 1.0, 1)
            else:
                return
            summary = dict(state)
        self.note(f"{device_id} used by hand -> {summary}")
        self.report(device_id)

    def tick(self) -> None:
        """Advance time: the AC moves the room temperature, motion clears."""
        now = time.monotonic()
        changed = []
        with self.lock:
            for device_id, kind in self.kinds.items():
                state = self.states[device_id]
                if kind == "thermostat" and _step_thermostat(state):
                    changed.append(device_id)
                elif kind == "motion_sensor" and state.get("motion") \
                        and now - self._motion_at.get(device_id, 0) >= MOTION_CLEAR_SECONDS:
                    state["motion"] = False
                    changed.append(device_id)
        for device_id in changed:
            self.report(device_id)


# ------------------------------------------------------------------ MQTT


def connect(devices: FakeDevices, client: mqtt.Client) -> None:
    def on_connect(client, userdata, flags, reason_code, properties=None):
        if reason_code != 0:
            devices.note(f"broker refused the connection: {reason_code}")
            return
        devices.note(f"connected to {settings.mqtt_host}:{settings.mqtt_port}")
        client.subscribe([(f"{BASE}/device/{d}/set", 1) for d in devices.states])
        devices.report_all()

    def on_message(client, userdata, msg):
        device_id = msg.topic.split("/")[2]
        try:
            command = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            devices.note(f"{device_id}: ignored a message that isn't JSON")
            return
        if isinstance(command, dict):
            devices.on_command(device_id, command)

    def on_disconnect(client, userdata, flags, reason_code, properties=None):
        devices.note(f"disconnected ({reason_code}); retrying...")

    if settings.mqtt_username:
        client.username_pw_set(settings.mqtt_username, settings.mqtt_password)
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    client.connect(settings.mqtt_host, settings.mqtt_port, keepalive=30)
    client.loop_start()


def load_devices(ids: list[str]) -> dict[str, tuple[str, dict]]:
    """(kind, last known state) for the requested devices, from the database."""
    from app.db import session_scope
    from app.db.models import Device

    with session_scope() as session:
        query = session.query(Device)
        rows = query.filter(Device.id.in_(ids)).all() if ids \
            else query.filter(Device.transport == "mqtt").all()
        found = {d.id: (d.kind, dict(d.state or {})) for d in rows if d.kind in KINDS}
        simulated = [d.id for d in rows if d.transport == "simulated"]

    missing = [i for i in ids if i not in found]
    if missing:
        sys.exit(f"Not registered (add them on the dashboard first): {', '.join(missing)}")
    if not found:
        sys.exit("No devices are set to 'real' yet. On the dashboard: Admin -> Devices -> "
                 "switch one to real (MQTT), or name devices: fake_device.py light.porch")
    if simulated:
        print(f"[fake-device] note: {', '.join(simulated)} still set to simulated on the "
              "dashboard -- the backend won't send it commands over MQTT until you switch it to real.")
    return found


# ------------------------------------------------------------------- GUI


def run_gui(devices: FakeDevices) -> None:
    import tkinter as tk

    bg, card, fg, muted = "#11161d", "#1a212b", "#e6edf3", "#8b98a5"
    root = tk.Tk()
    root.title("Fake smart-home devices (MQTT)")
    root.configure(bg=bg)
    tk.Label(root, text=f"Talking to the backend only through MQTT at "
                        f"{settings.mqtt_host}:{settings.mqtt_port}",
             bg=bg, fg=muted, font=("Segoe UI", 9)).pack(anchor="w", padx=12, pady=(10, 4))

    grid = tk.Frame(root, bg=bg)
    grid.pack(padx=8, pady=4)
    widgets: dict[str, dict[str, Any]] = {}
    labels = {"motion_sensor": "Walk past", "contact_sensor": "Open / close",
              "temperature_sensor": "Warm it up", "lock": "Turn key"}

    for i, (device_id, kind) in enumerate(devices.kinds.items()):
        box = tk.Frame(grid, bg=card, padx=10, pady=8)
        box.grid(row=i // 3, column=i % 3, padx=4, pady=4, sticky="nsew")
        tk.Label(box, text=device_id, bg=card, fg=fg, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(box, text=KINDS[kind].label, bg=card, fg=muted, font=("Segoe UI", 8)).pack(anchor="w")
        canvas = tk.Canvas(box, width=150, height=70, bg=card, highlightthickness=0)
        canvas.pack(pady=4)
        status = tk.Label(box, bg=card, fg=fg, font=("Consolas", 9))
        status.pack()
        if kind not in ("thermostat", "garage"):
            tk.Button(box, text=labels.get(kind, "Wall switch"), command=lambda d=device_id: devices.press(d),
                      bg="#243041", fg=fg, relief="flat", activebackground="#2d5a8c").pack(pady=(6, 0))
        widgets[device_id] = {"canvas": canvas, "status": status}

    log = tk.Text(root, height=8, width=70, bg="#0b0f14", fg=muted, relief="flat", font=("Consolas", 9))
    log.pack(fill="both", padx=12, pady=(4, 12))

    def draw(device_id: str, kind: str, s: dict) -> str:
        c = widgets[device_id]["canvas"]
        c.delete("all")
        if kind == "light":
            on, level = s.get("state") == "ON", int(s.get("brightness", 100))
            glow = _mix("#2a313b", "#ffd54a", level / 100 if on else 0)
            c.create_oval(55, 5, 95, 45, fill=glow, outline="#555")
            c.create_rectangle(66, 45, 84, 60, fill="#777", outline="")
            return f"{'ON ' + str(level) + '%' if on else 'OFF'}"
        if kind == "thermostat":
            mode = s.get("mode", "off")
            color = {"cool": "#4aa3ff", "heat": "#ff7b4a", "auto": "#b07bff"}.get(mode, "#555")
            c.create_text(75, 30, text=f"{s.get('current', 0):.1f}°F", fill=fg, font=("Segoe UI", 18, "bold"))
            c.create_text(75, 58, text=f"{mode.upper()} → {s.get('target')}°F", fill=color, font=("Segoe UI", 9))
            return ""
        if kind == "garage":
            frac = {"OPEN": 0.1, "CLOSED": 1.0}.get(s.get("state"), 0.5)
            c.create_rectangle(35, 5, 115, 65, outline="#777")
            c.create_rectangle(36, 6, 114, 6 + 58 * frac, fill="#5b6b7d", outline="")
            return s.get("state", "")
        if kind == "switch":
            on = s.get("state") == "ON"
            c.create_rectangle(50, 15, 100, 55, fill="#2f6b49" if on else "#3a2a2a", outline="")
            c.create_text(75, 35, text="ON" if on else "OFF", fill=fg, font=("Segoe UI", 12, "bold"))
            return ""
        if kind == "lock":
            locked = s.get("state") == "LOCKED"
            c.create_text(75, 35, text="🔒" if locked else "🔓", font=("Segoe UI Emoji", 26))
            return s.get("state", "")
        if kind == "motion_sensor":
            c.create_oval(60, 20, 90, 50, fill="#ff5c5c" if s.get("motion") else "#2a313b", outline="#555")
            return "MOTION" if s.get("motion") else "clear"
        if kind == "contact_sensor":
            open_ = s.get("contact") == "open"
            c.create_rectangle(55, 10, 95, 62, outline="#777")
            c.create_line(55, 10, 30 if open_ else 55, 20 if open_ else 62, fill=fg, width=3)
            return s.get("contact", "")
        if kind == "temperature_sensor":
            return f"{s.get('temperature')}°F  {s.get('humidity')}%"
        return json.dumps(s)

    shown = [0]

    def refresh():
        with devices.lock:
            snapshot = {d: dict(s) for d, s in devices.states.items()}
            lines = list(devices.log)
        for device_id, state in snapshot.items():
            widgets[device_id]["status"].configure(text=draw(device_id, devices.kinds[device_id], state))
        if len(lines) != shown[0]:
            log.delete("1.0", "end")
            log.insert("end", "\n".join(lines[-8:]))
            shown[0] = len(lines)
        root.after(200, refresh)

    def tick():
        devices.tick()
        root.after(TICK_SECONDS * 1000, tick)

    refresh()
    root.after(TICK_SECONDS * 1000, tick)
    root.mainloop()


def _mix(a: str, b: str, t: float) -> str:
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb))


# ------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description="Emulate smart-home devices over MQTT.")
    parser.add_argument("devices", nargs="*", help="device ids (default: every device set to 'real')")
    parser.add_argument("--no-gui", action="store_true", help="console only, no window")
    args = parser.parse_args()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"fake-devices-{int(time.time())}")
    devices = FakeDevices(
        load_devices(args.devices),
        lambda topic, payload: client.publish(topic, json.dumps(payload), qos=1,
                                              retain="/device/" in topic),
    )
    print(f"[fake-device] emulating: {', '.join(f'{d} ({k})' for d, k in devices.kinds.items())}")
    try:
        connect(devices, client)
    except OSError as exc:
        sys.exit(f"Can't reach the MQTT broker at {settings.mqtt_host}:{settings.mqtt_port} ({exc}).\n"
                 "Is Docker Desktop running? Then: docker compose up -d mosquitto")

    try:
        if args.no_gui:
            while True:
                time.sleep(TICK_SECONDS)
                devices.tick()
        else:
            run_gui(devices)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[fake-device] stopping")
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
