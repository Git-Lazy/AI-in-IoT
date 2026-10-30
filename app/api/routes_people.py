"""People: registration and face enrollment (MVP #4).

Adding people and faces is admin-only once an admin exists: enrolling a face is
what grants a person their role's access, so it can't be open to anyone.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app import auth
from app.api.schemas import PersonIn, PersonOut
from app.db import get_db
from app.db.models import FaceEmbedding, Person
from app.vision import cameras

router = APIRouter(prefix="/api/people", tags=["people"])


@router.get("", response_model=list[PersonOut])
def list_people(db: Session = Depends(get_db)):
    return db.query(Person).order_by(Person.name).all()


@router.post("", response_model=PersonOut, status_code=201)
def create_person(body: PersonIn, db: Session = Depends(get_db), _admin=Depends(auth.require_admin)):
    if db.query(Person).filter(Person.name == body.name).first():
        raise HTTPException(409, f"person '{body.name}' already exists")
    role = body.role if body.role != "admin" else "adult"   # admins are made via PATCH, with a password
    person = Person(name=body.name, is_household=body.is_household, role=role)
    db.add(person)
    db.commit()
    db.refresh(person)
    return person


@router.post("/{person_id}/faces", status_code=201)
async def enroll_face(
    person_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _admin=Depends(auth.require_admin),
):
    """Upload a photo, store only the embedding -- never the image itself."""
    import cv2

    person = _person(db, person_id)
    raw = np.frombuffer(await file.read(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "could not decode image")
    return _enroll_image(person, image, db)


@router.post("/{person_id}/faces/capture", status_code=201)
def capture_face(
    person_id: int,
    camera: str | None = None,
    db: Session = Depends(get_db),
    _admin=Depends(auth.require_admin),
):
    """Enroll from a live camera's current frame (the dashboard's Capture button).

    Uses a camera the pipeline already owns, so enrollment photos come from the
    exact camera that does the recognizing -- and the frame is never saved.
    """
    person = _person(db, person_id)
    pipeline = cameras.get(camera)
    if pipeline is None:
        raise HTTPException(503, "no camera running" if camera is None else f"no camera '{camera}'")
    frame = pipeline.latest_frame()
    if frame is None:
        raise HTTPException(503, "camera has no frame yet")
    return _enroll_image(person, frame, db)


@router.delete("/{person_id}", status_code=204)
def delete_person(person_id: int, db: Session = Depends(get_db), _admin=Depends(auth.require_admin)):
    person = _person(db, person_id)
    if person.role == "admin" and db.query(Person).filter(Person.role == "admin").count() <= 1:
        raise HTTPException(409, "can't delete the only admin")
    db.delete(person)
    db.commit()
    cameras.reload_galleries()


@router.delete("/{person_id}/faces", status_code=204)
def clear_faces(person_id: int, db: Session = Depends(get_db), _admin=Depends(auth.require_admin)):
    """Forget every enrolled face for this person (e.g. to re-enroll from scratch)."""
    person = _person(db, person_id)
    db.query(FaceEmbedding).filter(FaceEmbedding.person_id == person.id).delete()
    db.commit()
    cameras.reload_galleries()


def _person(db: Session, person_id: int) -> Person:
    person = db.get(Person, person_id)
    if person is None:
        raise HTTPException(404, "person not found")
    return person


def _enroll_image(person: Person, image: np.ndarray, db: Session) -> dict:
    """Find, align, embed, store, and refresh every camera's gallery."""
    recognizer = cameras.recognizer
    if recognizer is None:
        raise HTTPException(503, "vision subsystem is disabled")

    vector = recognizer.embed_image(image)
    if vector is None:
        raise HTTPException(
            422, "no clear face found -- face the camera, come a little closer, and hold still"
        )

    db.add(
        FaceEmbedding(
            person_id=person.id,
            vector=vector.astype(np.float32).tobytes(),
            dim=int(vector.shape[0]),
            model_name=recognizer.model_name,
        )
    )
    db.commit()
    cameras.reload_galleries()

    faces = (
        db.query(FaceEmbedding)
        .filter(FaceEmbedding.person_id == person.id,
                FaceEmbedding.model_name == recognizer.model_name)
        .count()
    )
    return {
        "person_id": person.id,
        "name": person.name,
        "embedding_dim": int(vector.shape[0]),
        "faces_enrolled": faces,
    }
