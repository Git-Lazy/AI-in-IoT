"""Admin authentication: password + face, and short-lived session tokens.

Logging in as admin takes two factors -- something you know (the password) and
something you are (your face, confirmed by a camera at that moment). A leaked
password alone isn't enough to rewrite who can open the garage.

Sessions are in memory, so a backend restart logs everyone out. That's fine
for a home hub and avoids storing tokens at rest.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.db.models import Person

PBKDF2_ROUNDS = 200_000


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ROUNDS)
    return f"{salt.hex()}${digest.hex()}"


def check_password(password: str, stored: str | None) -> bool:
    if not stored or "$" not in stored:
        return False
    salt_hex, digest_hex = stored.split("$", 1)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), PBKDF2_ROUNDS)
    return hmac.compare_digest(digest.hex(), digest_hex)


# ------------------------------------------------------------------- sessions


@dataclass
class AdminSession:
    person_id: int
    expires: float


_sessions: dict[str, AdminSession] = {}


def open_session(person_id: int) -> str:
    token = secrets.token_urlsafe(32)
    _sessions[token] = AdminSession(person_id, time.time() + settings.admin_session_hours * 3600)
    return token


def close_session(token: str) -> None:
    _sessions.pop(token, None)


def _token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    return header[7:] if header.lower().startswith("bearer ") else None


def session_admin(request: Request, db: Session) -> Person | None:
    """The admin behind this request's token, or None."""
    token = _token(request)
    session = _sessions.get(token) if token else None
    if session is None:
        return None
    if session.expires < time.time():
        close_session(token)
        return None
    person = db.get(Person, session.person_id)
    return person if person is not None and person.role == "admin" else None


def admin_exists(db: Session) -> bool:
    return db.query(Person).filter(Person.role == "admin", Person.password_hash.isnot(None)).count() > 0


def require_admin(request: Request, db: Session = Depends(get_db)) -> Person | None:
    """FastAPI dependency for admin-only endpoints.

    Until the first admin is created the house is in setup mode and these
    endpoints are open -- you have to be able to enroll the admin's face before
    there's an admin to log in with. Returns None in setup mode.
    """
    if not admin_exists(db):
        return None
    admin = session_admin(request, db)
    if admin is None:
        raise HTTPException(401, "admin login required")
    return admin
