"""Event history and notifications (MVP #6, #7)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.schemas import EventOut
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
