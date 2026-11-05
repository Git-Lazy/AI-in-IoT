"""Event history and notifications (MVP #6, #7)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.schemas import EventOut
from app.auth import require_admin
from app.config import settings
from app.db import get_db
from app.db.models import EventLog
from app.notifications import notifier

router = APIRouter(prefix="/api", tags=["events"])


@router.get("/events", response_model=list[EventOut])
def list_events(
    limit: int = Query(50, ge=1, le=500),
    type: str | None = None,
    db: Session = Depends(get_db),
):
    query = db.query(EventLog)
    if type:
        query = query.filter(EventLog.type == type)
    return query.order_by(EventLog.timestamp.desc()).limit(limit).all()


@router.get("/notifications")
def list_notifications(limit: int = Query(20, ge=1, le=100)):
    return notifier.recent(limit)


@router.get("/notifications/status")
def notification_status():
    return {
        "phone": notifier.phone_enabled,
        "server": settings.ntfy_server,
        "min_priority": settings.ntfy_min_priority,
    }


@router.post("/notifications/test", dependencies=[Depends(require_admin)])
async def test_notification():
    """Push one notification straight to the phone, bypassing the cooldown."""
    if not notifier.phone_enabled:
        raise HTTPException(400, "Phone notifications are off: set NTFY_TOPIC in .env and restart.")
    message = "Test notification -- phone alerts are working."
    if not await notifier.push(message, "normal", title="Home AI test"):
        raise HTTPException(502, "Push failed -- check the backend log and your NTFY settings.")
    return {"sent": True}
