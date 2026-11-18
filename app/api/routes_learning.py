"""Learned preferences: training examples, the trained models, and what they'd do."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import auth
from app.db import get_db
from app.db.models import Device, Person, PreferenceSample
from app.learning import LEAVE, demo
from app.learning.model import learner
from app.learning.samples import describe, features_at

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


@router.get("/models")
def models(person_id: int | None = None):
    """Every trained model: accuracy vs. the "do the usual" baseline, and its rules."""
    return {
        "trained_at": learner.trained_at.isoformat() if learner.trained_at else None,
        "models": [m.to_dict() for m in learner.models(person_id)],
    }


@router.post("/train")
def train(_admin=Depends(auth.require_admin)):
    return {"models": len(learner.train())}


@router.get("/predict/{person_id}")
def predict(
    person_id: int, db: Session = Depends(get_db),
    hour: float | None = None, weekend: bool | None = None,
    temp_f: float | None = None, others_home: int = 0,
):
    """What the house would do if this person walked in now -- or in the
    situation given by the query (try ?hour=19&temp_f=84)."""
    if db.get(Person, person_id) is None:
        raise HTTPException(404, "no such person")
    features = features_at(datetime.now().astimezone(), temp_f, others_home)
    if hour is not None:
        features["hour"] = hour
    if weekend is not None:
        features["weekend"] = int(weekend)
    return {"features": features, "predictions": [p.to_dict() for p in learner.predict(person_id, features)]}


class DemoHistoryIn(BaseModel):
    person_id: int | None = None          # default: the logged-in admin
    weeks: int = Field(6, ge=1, le=26)


@router.post("/demo-history", status_code=201)
def add_demo_history(body: DemoHistoryIn, db: Session = Depends(get_db), admin=Depends(auth.require_admin)):
    """Generate weeks of arrivals that follow made-up habits, then retrain."""
    person_id = body.person_id or (admin.id if admin else None)
    if person_id is None or db.get(Person, person_id) is None:
        raise HTTPException(400, "say whose history to generate (person_id)")
    try:
        added = demo.seed(person_id, body.weeks)
    except LookupError as exc:
        raise HTTPException(400, str(exc)) from None
    return {"added": added, "models": len(learner.train())}


@router.delete("/demo-history")
def remove_demo_history(_admin=Depends(auth.require_admin)):
    removed = demo.clear()
    return {"removed": removed, "models": len(learner.train())}
