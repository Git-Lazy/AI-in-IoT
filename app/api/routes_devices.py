"""Devices: registry and manual control (MVP #1, #9)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.schemas import CommandIn, DeviceIn, DeviceOut
from app.db import get_db
from app.db.models import Device
from app.events import Event, EventType, bus
from app.iot import mqtt_bridge

router = APIRouter(prefix="/api/devices", tags=["devices"])


@router.get("", response_model=list[DeviceOut])
def list_devices(db: Session = Depends(get_db)):
    return db.query(Device).order_by(Device.room, Device.name).all()


@router.post("", response_model=DeviceOut, status_code=201)
def register_device(body: DeviceIn, db: Session = Depends(get_db)):
    if db.get(Device, body.id):
        raise HTTPException(409, f"device '{body.id}' already registered")
    device = Device(id=body.id, name=body.name, kind=body.kind, room=body.room, state={})
    db.add(device)
    db.commit()
    db.refresh(device)
    return device


@router.post("/{device_id}/command")
async def send_command(device_id: str, body: CommandIn, db: Session = Depends(get_db)):
    if db.get(Device, device_id) is None:
        raise HTTPException(404, "device not registered")

    delivered = mqtt_bridge.command(device_id, body.payload)
    await bus.publish(
        Event(
            type=EventType.DEVICE_COMMAND,
            source="api",
            device_id=device_id,
            payload={**body.payload, "delivered": delivered},
        )
    )
    return {"device_id": device_id, "payload": body.payload, "delivered": delivered}
