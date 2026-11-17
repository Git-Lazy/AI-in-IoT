"""Learned preferences: the training examples collected so far."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.db.models import Device, Person, PreferenceSample
from app.learning import LEAVE
from app.learning.samples import describe

router = APIRouter(prefix="/api/learning", tags=["learning"])


@router.get("/samples")
def samples(limit: int = 50, db: Session = Depends(get_db)):
    """Recent arrivals as training examples, newest first, plus totals per person."""
    names = {p.id: p.name for p in db.query(Person)}
    kinds = {d.id: (d.kind, d.name) for d in db.query(Device)}
    rows = db.query(PreferenceSample).order_by(PreferenceSample.arrived_at.desc())

    totals: dict[str, dict[str, int]] = {}
    for s in rows:
        t = totals.setdefault(names.get(s.person_id, "?"), {"complete": 0, "open": 0})
        t["open" if s.labels is None else "complete"] += 1

    recent = []
    for s in rows.limit(max(1, min(limit, 500))):
        changed = {
            kinds[d][1]: describe(kinds[d][0], v)
            for d, v in (s.labels or {}).items()
            if v != LEAVE and d in kinds
        }
        recent.append({
            "id": s.id,
            "person": names.get(s.person_id, "?"),
            "arrived_at": s.arrived_at.isoformat(),
            "features": s.features,
            "status": "open" if s.labels is None else "complete",
            "changed": changed,
            "source": s.source,
        })
    return {"totals": totals, "recent": recent}
