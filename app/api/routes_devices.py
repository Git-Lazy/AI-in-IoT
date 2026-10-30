"""Devices: registry, simulation, and access-checked control (MVP #1, #9).

Who is allowed to press a button is decided by a camera, not a login. A command
from the dashboard is checked against whoever the camera in that device's room
can see right now: a child standing at the camera can switch the lights, but the
garage stays shut. A logged-in admin skips the camera check.
"""

from __future__ import annotations

import asyncio
import re
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app import access, auth
from app.api.schemas import CommandIn, DeviceIn, DeviceOut, DeviceUpdate
from app.db import get_db
from app.db.models import AccessRule, Device, Person, Preference
from app.events import Event, EventType, bus
from app.iot import devices
from app.iot.simulator import simulator
from app.notifications import notifier
from app.vision import cameras

router = APIRouter(prefix="/api/devices", tags=["devices"])

# How long a command waits for the camera to confirm who's there.
FACE_WAIT_SECONDS = 3.0

DEMO_DEVICES = [
    ("light.entry", "Entry Light", "light", "Entry"),
    ("light.living_room", "Living Room Light", "light", "Living Room"),
    ("thermostat.main", "AC", "thermostat", "Living Room"),
    ("lock.front_door", "Front Door Lock", "lock", "Entry"),
    ("garage.main", "Garage Door", "garage", "Garage"),
    ("switch.tv", "TV", "switch", "Living Room"),
    ("sensor.entry_motion", "Entry Motion", "motion_sensor", "Entry"),
    ("sensor.front_door", "Front Door Sensor", "contact_sensor", "Entry"),
    ("sensor.living_room_temp", "Living Room Temperature", "temperature_sensor", "Living Room"),
]


@router.get("", response_model=list[DeviceOut])
def list_devices(db: Session = Depends(get_db)):
    return db.query(Device).order_by(Device.room, Device.name).all()


@router.get("/kinds")
def kinds():
    return {k: {"label": v.label, "controllable": v.controllable} for k, v in devices.KINDS.items()}


# ------------------------------------------------------------- admin: registry


@router.post("", response_model=DeviceOut, status_code=201)
async def add_device(body: DeviceIn, db: Session = Depends(get_db), _admin=Depends(auth.require_admin)):
    if body.kind not in devices.KINDS:
        raise HTTPException(400, f"unknown kind '{body.kind}'")
    device_id = body.id or _make_id(db, body.kind, body.name)
    if db.get(Device, device_id):
        raise HTTPException(409, f"device '{device_id}' already registered")
    device = Device(
        id=device_id, name=body.name, kind=body.kind, room=body.room,
        transport=body.transport, state={},
    )
    db.add(device)
    db.commit()
    if device.transport == "simulated":
        await simulator.add(device.id, device.kind)
    db.refresh(device)
    return device


@router.post("/demo", status_code=201)
async def add_demo_devices(db: Session = Depends(get_db), _admin=Depends(auth.require_admin)):
    """A simulated starter house: lights, AC, lock, garage, TV, sensors."""
    added = []
    for device_id, name, kind, room in DEMO_DEVICES:
        if db.get(Device, device_id):
            continue
        db.add(Device(id=device_id, name=name, kind=kind, room=room, transport="simulated", state={}))
        added.append((device_id, kind))
    db.commit()
    for device_id, kind in added:
        await simulator.add(device_id, kind)
    return {"added": [d for d, _ in added]}


@router.patch("/{device_id}", response_model=DeviceOut)
async def update_device(
    device_id: str, body: DeviceUpdate, db: Session = Depends(get_db),
    _admin=Depends(auth.require_admin),
):
    device = _device(db, device_id)
    if body.name is not None:
        device.name = body.name
    if body.room is not None:
        device.room = body.room
    if body.transport is not None and body.transport != device.transport:
        # Swap simulation for the real device (or back) -- state carries over.
        device.transport = body.transport
        device.online = False
        if body.transport == "simulated":
            await simulator.add(device.id, device.kind, device.state)
        else:
            simulator.remove(device.id)
    db.commit()
    db.refresh(device)
    return device


@router.delete("/{device_id}", status_code=204)
def remove_device(device_id: str, db: Session = Depends(get_db), _admin=Depends(auth.require_admin)):
    device = _device(db, device_id)
    simulator.remove(device.id)
    db.query(Preference).filter(Preference.device_id == device.id).delete()
    db.query(AccessRule).filter(AccessRule.target == access.device_target(device.id)).delete()
    db.delete(device)
    db.commit()


@router.post("/{device_id}/simulate")
async def simulate_sensor(
    device_id: str, body: dict | None = None, db: Session = Depends(get_db),
    _admin=Depends(auth.require_admin),
):
    """Fake a sensor reading: motion, a door opening, a temperature."""
    device = _device(db, device_id)
    if device.transport != "simulated":
        raise HTTPException(400, "only simulated devices can be triggered")
    try:
        state = await simulator.trigger(device.id, body or {})
    except (LookupError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from None
    return {"device_id": device.id, "state": state}


# ------------------------------------------------------------------ control


@router.post("/{device_id}/command")
async def send_command(
    device_id: str, body: CommandIn, request: Request, db: Session = Depends(get_db)
):
    device = _device(db, device_id)

    admin = auth.session_admin(request, db)
    if admin is not None:
        actor, via = admin.name, "admin"
    elif not auth.admin_exists(db):
        actor, via = None, "setup mode"
    else:
        actor, via = await _camera_check(db, device), "camera"

    try:
        result = await devices.send(device.id, body.payload, source=f"api:{via}", actor=actor)
    except devices.CommandError as exc:
        raise HTTPException(400, str(exc)) from None
    return {**result, "by": actor, "via": via}


async def _camera_check(db: Session, device: Device) -> str:
    """Name of the person at the device's camera who may use it, or raise 403."""
    camera = cameras.for_room(device.room)
    if camera is None:
        raise HTTPException(403, "No camera is running to check who you are. Log in as admin instead.")

    deadline = time.monotonic() + FACE_WAIT_SECONDS
    present = camera.present_people()
    while not present and time.monotonic() < deadline:
        await asyncio.sleep(0.25)
        present = camera.present_people()

    if not present:
        await _deny(device, camera.camera_name, None, "nobody recognized")
        raise HTTPException(
            403, f"Nobody recognized at the {camera.camera_name} camera. Face it and try again."
        )

    refusals = []
    for match in present:
        person = db.get(Person, match.person_id)
        if person is None:
            continue
        allowed, why = access.can_control(db, person, device)
        if allowed:
            await bus.publish(Event(
                type=EventType.ACCESS_GRANTED, source=f"vision:{camera.camera_name}",
                person_id=person.id, device_id=device.id,
                payload={"name": person.name, "device": device.name, "reason": why},
            ))
            return person.name
        refusals.append((person, why))

    person, why = refusals[0] if refusals else (None, "not registered")
    await _deny(device, camera.camera_name, person, why)
    who = person.name if person else "You"
    raise HTTPException(403, f"{who} isn't allowed to use {device.name} ({why}).")


async def _deny(device: Device, camera: str, person: Person | None, why: str) -> None:
    name = person.name if person else "someone"
    await bus.publish(Event(
        type=EventType.ACCESS_DENIED, source=f"vision:{camera}",
        person_id=person.id if person else None, device_id=device.id,
        payload={"name": name, "device": device.name, "reason": why},
    ))
    if person is not None:
        notifier.send(f"{person.name} tried to use {device.name} -- blocked ({why}).", priority="high")


# ---------------------------------------------------------------- helpers


def _device(db: Session, device_id: str) -> Device:
    device = db.get(Device, device_id)
    if device is None:
        raise HTTPException(404, "device not registered")
    return device


def _make_id(db: Session, kind: str, name: str) -> str:
    prefix = {"motion_sensor": "sensor", "contact_sensor": "sensor", "temperature_sensor": "sensor"}.get(kind, kind)
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "device"
    base = candidate = f"{prefix}.{slug}"
    n = 2
    while db.get(Device, candidate):
        candidate = f"{base}_{n}"
        n += 1
    return candidate
