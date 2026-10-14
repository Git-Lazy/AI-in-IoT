"""People: registration and face enrollment (MVP #4)."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from app.api.schemas import PersonIn, PersonOut
from app.db import get_db
from app.db.models import FaceEmbedding, Person

router = APIRouter(prefix="/api/people", tags=["people"])


@router.get("", response_model=list[PersonOut])
def list_people(db: Session = Depends(get_db)):
    return db.query(Person).order_by(Person.name).all()


@router.post("", response_model=PersonOut, status_code=201)
def create_person(body: PersonIn, db: Session = Depends(get_db)):
    if db.query(Person).filter(Person.name == body.name).first():
        raise HTTPException(409, f"person '{body.name}' already exists")
    person = Person(name=body.name, is_household=body.is_household)
    db.add(person)
    db.commit()
    db.refresh(person)
    return person


@router.post("/{person_id}/faces", status_code=201)
async def enroll_face(
    person_id: int,
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload a photo, store only the embedding -- never the image itself."""
    import cv2

    person = db.get(Person, person_id)
    if person is None:
        raise HTTPException(404, "person not found")

    raw = np.frombuffer(await file.read(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "could not decode image")

    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(503, "vision subsystem is disabled")

    vector = pipeline._encoder.embed(image)
    if vector is None:
        raise HTTPException(422, "no usable face found in image")

    db.add(
        FaceEmbedding(
            person_id=person.id,
            vector=vector.astype(np.float32).tobytes(),
            dim=int(vector.shape[0]),
            model_name=pipeline._encoder.name,
        )
    )
    db.commit()
    pipeline.reload_gallery()

    return {"person_id": person.id, "name": person.name, "embedding_dim": int(vector.shape[0])}


@router.delete("/{person_id}", status_code=204)
def delete_person(person_id: int, request: Request, db: Session = Depends(get_db)):
    person = db.get(Person, person_id)
    if person is None:
        raise HTTPException(404, "person not found")
    db.delete(person)
    db.commit()
    if request.app.state.pipeline is not None:
        request.app.state.pipeline.reload_gallery()
