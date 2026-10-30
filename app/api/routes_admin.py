"""Admin: first-run setup, login, roles, permissions, and arrival preferences."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app import access, auth
from app.api.schemas import AccessRuleIn, AdminLoginIn, AdminSetupIn, PersonUpdate, PreferenceIn
from app.config import settings
from app.db import get_db
from app.db.models import AccessRule, Device, FaceEmbedding, Person, Preference
from app.events import Event, EventType, bus
from app.iot import devices
from app.vision import cameras

router = APIRouter(prefix="/api/admin", tags=["admin"])

# How long login waits for the admin's face to be confirmed by a camera.
FACE_WAIT_SECONDS = 6.0


@router.get("/status")
def status(request: Request, db: Session = Depends(get_db)):
    admin = auth.session_admin(request, db)
    return {
        "setup_required": not auth.admin_exists(db),
        "admin": {"id": admin.id, "name": admin.name} if admin else None,
        "face_verify": settings.admin_face_verify,
    }


@router.post("/setup")
def setup(body: AdminSetupIn, db: Session = Depends(get_db)):
    """Create the first admin. Only works while no admin exists."""
    if auth.admin_exists(db):
        raise HTTPException(409, "an admin already exists -- log in instead")
    person = db.query(Person).filter(Person.name == body.name).one_or_none()
    if person is None:
        person = Person(name=body.name)
        db.add(person)
    person.role = "admin"
    person.password_hash = auth.hash_password(body.password)
    db.commit()
    return {"token": auth.open_session(person.id), "name": person.name, "warnings": []}


@router.post("/login")
async def login(body: AdminLoginIn, db: Session = Depends(get_db)):
    person = db.query(Person).filter(Person.name == body.name).one_or_none()
    if person is None or person.role != "admin" or not auth.check_password(body.password, person.password_hash):
        raise HTTPException(401, "wrong name or password")

    warnings: list[str] = []
    if settings.admin_face_verify:
        has_face = (
            cameras.recognizer is not None
            and db.query(FaceEmbedding)
            .filter(FaceEmbedding.person_id == person.id,
                    FaceEmbedding.model_name == cameras.recognizer.model_name)
            .count() > 0
        )
        if not cameras.running:
            warnings.append("No camera running -- logged in with password only.")
        elif not has_face:
            warnings.append("You have no face enrolled -- logged in with password only. Enroll one.")
        elif not await _wait_for_face(person.id):
            raise HTTPException(
                403, "Password OK, but your face wasn't confirmed. Look at the camera and try again."
            )

    await bus.publish(
        Event(
            type=EventType.ADMIN_LOGIN,
            source="api",
            person_id=person.id,
            payload={"name": person.name, "face_verified": not warnings},
        )
    )
    return {"token": auth.open_session(person.id), "name": person.name, "warnings": warnings}


async def _wait_for_face(person_id: int) -> bool:
    deadline = time.monotonic() + FACE_WAIT_SECONDS
    while time.monotonic() < deadline:
        if any(m.person_id == person_id for m, _ in cameras.present_anywhere()):
            return True
        await asyncio.sleep(0.25)
    return False


@router.post("/logout")
def logout(request: Request):
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        auth.close_session(header[7:])
    return {"ok": True}


# ------------------------------------------------------------------ people


@router.get("/roles")
def roles():
    return {
        "roles": access.ROLES,
        "defaults": access.ROLE_DEFAULTS,
        "kinds": {k: devices.KINDS[k].label for k in access.CONTROLLABLE},
    }


@router.get("/people")
def people(db: Session = Depends(get_db), _admin=Depends(auth.require_admin)):
    out = []
    model = cameras.recognizer.model_name if cameras.recognizer else None
    for person in db.query(Person).order_by(Person.name):
        out.append({
            "id": person.id,
            "name": person.name,
            "role": person.role,
            "has_password": person.password_hash is not None,
            "faces": sum(1 for f in person.faces if model is None or f.model_name == model),
            "permissions": access.summary(db, person),
            "rules": {r.target: r.allowed for r in person.access_rules},
            "preferences": {p.device_id: p.payload for p in person.preferences},
        })
    return out


@router.patch("/people/{person_id}")
def update_person(
    person_id: int, body: PersonUpdate, db: Session = Depends(get_db),
    _admin=Depends(auth.require_admin),
):
    person = _person(db, person_id)
    if body.role is not None and body.role != person.role:
        if person.role == "admin" and _admin_count(db) <= 1:
            raise HTTPException(409, "can't demote the only admin")
        person.role = body.role
        if body.role != "admin":
            person.password_hash = None
    if body.password is not None:
        if person.role != "admin":
            raise HTTPException(400, "only admins have passwords")
        person.password_hash = auth.hash_password(body.password)
    if person.role == "admin" and person.password_hash is None:
        raise HTTPException(400, "a new admin needs a password")
    db.commit()
    return {"id": person.id, "role": person.role}


@router.put("/people/{person_id}/access")
def set_access(
    person_id: int, body: AccessRuleIn, db: Session = Depends(get_db),
    _admin=Depends(auth.require_admin),
):
    person = _person(db, person_id)
    rule = (
        db.query(AccessRule)
        .filter(AccessRule.person_id == person.id, AccessRule.target == body.target)
        .one_or_none()
    )
    if body.allowed is None:
        if rule is not None:
            db.delete(rule)
    elif rule is None:
        db.add(AccessRule(person_id=person.id, target=body.target, allowed=body.allowed))
    else:
        rule.allowed = body.allowed
    db.commit()
    return {"permissions": access.summary(db, person)}


@router.put("/people/{person_id}/preferences/{device_id}")
def set_preference(
    person_id: int, device_id: str, body: PreferenceIn, db: Session = Depends(get_db),
    _admin=Depends(auth.require_admin),
):
    person = _person(db, person_id)
    device = db.get(Device, device_id)
    if device is None:
        raise HTTPException(404, "device not registered")
    try:
        payload = devices.normalize(device.kind, body.payload)
    except devices.CommandError as exc:
        raise HTTPException(400, str(exc)) from None

    allowed, why = access.can_control(db, person, device)
    pref = (
        db.query(Preference)
        .filter(Preference.person_id == person.id, Preference.device_id == device_id)
        .one_or_none()
    )
    if pref is None:
        db.add(Preference(person_id=person.id, device_id=device_id, payload=payload))
    else:
        pref.payload = payload
    db.commit()
    warning = None if allowed else f"{person.name} isn't allowed to use {device.name} ({why}), so this won't apply."
    return {"device_id": device_id, "payload": payload, "warning": warning}


@router.delete("/people/{person_id}/preferences/{device_id}", status_code=204)
def delete_preference(
    person_id: int, device_id: str, db: Session = Depends(get_db),
    _admin=Depends(auth.require_admin),
):
    db.query(Preference).filter(
        Preference.person_id == person_id, Preference.device_id == device_id
    ).delete()
    db.commit()


def _person(db: Session, person_id: int) -> Person:
    person = db.get(Person, person_id)
    if person is None:
        raise HTTPException(404, "person not found")
    return person


def _admin_count(db: Session) -> int:
    return db.query(Person).filter(Person.role == "admin").count()
