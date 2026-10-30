"""Request/response models for the API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PersonIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    is_household: bool = True
    role: Literal["admin", "adult", "child", "guest"] = "adult"


class PersonOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    is_household: bool
    role: str
    is_home: bool
    last_seen: datetime | None
    created_at: datetime


class DeviceIn(BaseModel):
    # Optional: generated from kind + name ("light.living_room") when omitted.
    id: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[a-z0-9_.\-]+$")
    name: str = Field(min_length=1, max_length=80)
    kind: str = "light"
    room: str = ""
    transport: Literal["simulated", "mqtt"] = "simulated"


class DeviceUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    room: str | None = None
    transport: Literal["simulated", "mqtt"] | None = None


class DeviceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    kind: str
    room: str
    transport: str
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


# --------------------------------------------------------------------- admin


class AdminSetupIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=6, max_length=200)


class AdminLoginIn(BaseModel):
    name: str
    password: str


class PersonUpdate(BaseModel):
    role: Literal["admin", "adult", "child", "guest"] | None = None
    password: str | None = Field(default=None, min_length=6, max_length=200)


class AccessRuleIn(BaseModel):
    """target: "kind:light" or "device:garage.main". allowed=None resets to default."""

    target: str = Field(pattern=r"^(kind|device):[A-Za-z0-9_.\-]+$")
    allowed: bool | None


class PreferenceIn(BaseModel):
    payload: dict[str, Any]
