"""Who may control what.

Each person has a role, and each role has a default yes/no per device kind. An
admin can then add exceptions for one person: a whole kind ("Sam may use the
AC") or one device ("Sam may not use the garage"). Lookup order, most specific
first:

    device rule  >  kind rule  >  role default

Admins can control everything and can't be restricted -- otherwise an admin
could lock themselves out of the settings that would undo it.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import AccessRule, Device, Person
from app.iot.devices import KINDS

ROLES = ("admin", "adult", "child", "guest")

CONTROLLABLE = tuple(k for k, spec in KINDS.items() if spec.controllable)

ROLE_DEFAULTS: dict[str, dict[str, bool]] = {
    "admin": {k: True for k in CONTROLLABLE},
    "adult": {k: True for k in CONTROLLABLE},
    "child": {"light": True, "switch": True, "thermostat": False, "lock": False, "garage": False},
    "guest": {"light": True, "switch": False, "thermostat": False, "lock": False, "garage": False},
}


def kind_target(kind: str) -> str:
    return f"kind:{kind}"


def device_target(device_id: str) -> str:
    return f"device:{device_id}"


def can_control(session: Session, person: Person, device: Device) -> tuple[bool, str]:
    """(allowed, why) -- the reason is shown to the user and logged."""
    if person.role == "admin":
        return True, "admin"
    if device.kind not in CONTROLLABLE:
        return False, f"{device.kind} can't be controlled"

    rules = {
        r.target: r.allowed
        for r in session.query(AccessRule).filter(AccessRule.person_id == person.id)
    }
    if device_target(device.id) in rules:
        allowed = rules[device_target(device.id)]
        return allowed, f"{'allowed' if allowed else 'blocked'} for {device.name} by admin"
    if kind_target(device.kind) in rules:
        allowed = rules[kind_target(device.kind)]
        return allowed, f"{'allowed' if allowed else 'blocked'} for every {device.kind} by admin"

    allowed = ROLE_DEFAULTS.get(person.role, {}).get(device.kind, False)
    return allowed, f"{person.role} default"


def summary(session: Session, person: Person) -> dict[str, bool]:
    """Effective kind-level permissions, for the admin screen."""
    out = dict(ROLE_DEFAULTS.get(person.role, {}))
    for rule in session.query(AccessRule).filter(AccessRule.person_id == person.id):
        if rule.target.startswith("kind:"):
            out[rule.target.removeprefix("kind:")] = rule.allowed
    if person.role == "admin":
        out = {k: True for k in CONTROLLABLE}
    return out
