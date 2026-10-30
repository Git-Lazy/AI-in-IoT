"""Database schema: people, faces, access rules, preferences, devices, event log."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Boolean,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Person(Base):
    """A registered household member."""

    __tablename__ = "people"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    is_household: Mapped[bool] = mapped_column(Boolean, default=True)
    # admin | adult | child | guest -- see app/access.py for what each may do.
    role: Mapped[str] = mapped_column(String(16), default="adult")
    # Only admins have one. PBKDF2, "salt$hash" hex -- see app/auth.py.
    password_hash: Mapped[str | None] = mapped_column(String(200), nullable=True)
    # Presence state, updated by the rules engine on arrive/leave events.
    is_home: Mapped[bool] = mapped_column(Boolean, default=False)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    faces: Mapped[list["FaceEmbedding"]] = relationship(
        back_populates="person", cascade="all, delete-orphan"
    )
    access_rules: Mapped[list["AccessRule"]] = relationship(cascade="all, delete-orphan")
    preferences: Mapped[list["Preference"]] = relationship(cascade="all, delete-orphan")


class FaceEmbedding(Base):
    """One enrolled face vector.

    We store the embedding, not the photograph -- a privacy property that comes
    straight out of the FaceNet approach (Schroff et al., 2015).
    """

    __tablename__ = "face_embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"))
    vector: Mapped[bytes] = mapped_column(LargeBinary)       # float32 array, .tobytes()
    dim: Mapped[int] = mapped_column(Integer, default=512)
    model_name: Mapped[str] = mapped_column(String(50), default="stub")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    person: Mapped[Person] = relationship(back_populates="faces")


class Device(Base):
    """An IoT device the platform can observe or control."""

    __tablename__ = "devices"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)   # e.g. "light.living_room"
    name: Mapped[str] = mapped_column(String(80))
    kind: Mapped[str] = mapped_column(String(32))                   # see app/iot/devices.py KINDS
    room: Mapped[str] = mapped_column(String(50), default="")
    # "simulated" runs inside the backend; "mqtt" is real hardware on the broker.
    # Flip one to the other when the physical device arrives -- nothing else changes.
    transport: Mapped[str] = mapped_column(String(16), default="simulated")
    state: Mapped[dict] = mapped_column(JSON, default=dict)         # last known state
    online: Mapped[bool] = mapped_column(Boolean, default=False)
    last_update: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AccessRule(Base):
    """A per-person exception to their role's defaults.

    `target` is "kind:<kind>" (every device of that kind) or "device:<id>" (one
    device). A device rule beats a kind rule, which beats the role default.
    """

    __tablename__ = "access_rules"
    __table_args__ = (UniqueConstraint("person_id", "target"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"))
    target: Mapped[str] = mapped_column(String(80))
    allowed: Mapped[bool] = mapped_column(Boolean)


class Preference(Base):
    """What a person wants a device set to when they arrive home."""

    __tablename__ = "preferences"
    __table_args__ = (UniqueConstraint("person_id", "device_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"))
    device_id: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JSON, default=dict)   # same shape as a command


class EventLog(Base):
    """Durable record of every event on the bus -- MVP requirement #6."""

    __tablename__ = "event_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    type: Mapped[str] = mapped_column(String(40), index=True)
    source: Mapped[str] = mapped_column(String(80))
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    person_id: Mapped[int | None] = mapped_column(ForeignKey("people.id"), nullable=True)
    device_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
