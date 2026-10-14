"""Request/response models for the API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PersonIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    is_household: bool = True


class PersonOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    is_household: bool
    is_home: bool
    last_seen: datetime | None
    created_at: datetime


class DeviceIn(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    name: str
    kind: str = "light"
    room: str = ""


class DeviceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    kind: str
    room: str
    state: dict[str, Any]
    online: bool
    last_update: datetime | None


class CommandIn(BaseModel):
    """Arbitrary JSON command forwarded to the device over MQTT."""

    payload: dict[str, Any] = Field(default_factory=lambda: {"state": "ON"})


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    type: str
    source: str
    payload: dict[str, Any]
    confidence: float | None
    person_id: int | None
    device_id: str | None
    timestamp: datetime


class HealthOut(BaseModel):
    status: str
    version: str
    mqtt_connected: bool
    vision_running: bool
    detector_backend: str
    recognizer_backend: str
