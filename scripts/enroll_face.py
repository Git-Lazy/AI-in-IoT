"""Enroll a household member from a folder of photos.

    python scripts/enroll_face.py --name "Samuel" --images data/faces/samuel

Stores embeddings only; the source photos are never written to the database.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.db import init_db, session_scope  # noqa: E402
from app.db.models import FaceEmbedding, Person  # noqa: E402
from app.vision.recognizer import FaceRecognizer  # noqa: E402

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Enroll a face into the gallery")
    parser.add_argument("--name", required=True)
    parser.add_argument("--images", required=True, type=Path, help="file or directory")
    args = parser.parse_args()

    paths = (
        [p for p in sorted(args.images.iterdir()) if p.suffix.lower() in SUFFIXES]
        if args.images.is_dir()
        else [args.images]
    )
    if not paths:
        print(f"no images found in {args.images}")
        return 1

    init_db()
    recognizer = FaceRecognizer()
    stored = 0

    with session_scope() as session:
        person = session.query(Person).filter(Person.name == args.name).one_or_none()
        if person is None:
            person = Person(name=args.name)
            session.add(person)
            session.flush()
            print(f"created person '{args.name}' (id={person.id})")

        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                print(f"  skip {path.name}: unreadable")
                continue
            # Same find -> align -> embed path the live pipeline uses.
            vector = recognizer.embed_image(image)
            if vector is None:
                print(f"  skip {path.name}: no clear face (too small, blurry, or turned away)")
                continue
            session.add(
                FaceEmbedding(
                    person_id=person.id,
                    vector=vector.astype(np.float32).tobytes(),
                    dim=int(vector.shape[0]),
                    model_name=recognizer.model_name,
                )
            )
            stored += 1
            print(f"  + {path.name}")

    print(f"\nstored {stored} embedding(s) for '{args.name}' using {recognizer.model_name}")
    print("restart the backend (or POST to /api/people) to refresh the live gallery")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
