"""Face recognition: turn a face crop into an embedding, then match it to a person.

Matching is cosine distance against enrolled embeddings loaded from the database.
That part is backend-independent -- only `embed()` changes when FaceNet replaces
the stub.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from app.config import settings

log = logging.getLogger(__name__)


@dataclass
class Match:
    person_id: int | None     # None => unknown face
    name: str
    distance: float           # cosine distance; lower is a better match


class FaceCropper:
    """Finds the largest face in an image with OpenCV's Haar cascade.

    Enrollment and the live pipeline must both embed the *same kind* of image --
    a tight face crop -- or gallery vectors and live vectors aren't comparable and
    every match fails. Haar is weak but free; swapping in MTCNN once
    facenet-pytorch is installed is a change to this one class.

    Not thread-safe: give each thread its own instance.
    """

    def __init__(self) -> None:
        import cv2

        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)

    def crop(self, image_bgr: np.ndarray) -> np.ndarray | None:
        if image_bgr.size == 0:
            return None
        import cv2

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        return image_bgr[fy : fy + fh, fx : fx + fw]


class FaceEncoder(ABC):
    """Produces a normalized embedding vector for a face crop."""

    dim: int = 512
    name: str = "base"

    @abstractmethod
    def embed(self, face_bgr: np.ndarray) -> np.ndarray | None:
        """Return a unit-length float32 vector, or None if no usable face."""


class StubEncoder(FaceEncoder):
    """Deterministic pseudo-embedding from pixel statistics.

    Not real recognition -- but it is stable for the same image, so the enrollment
    flow, the database round-trip, and the matching math can all be tested before
    FaceNet exists.
    """

    dim = 64
    name = "stub"

    def embed(self, face_bgr: np.ndarray) -> np.ndarray | None:
        if face_bgr.size == 0:
            return None
        import cv2

        small = cv2.resize(face_bgr, (8, 8))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32).flatten()
        norm = np.linalg.norm(gray)
        return gray / norm if norm > 0 else None


class FaceNetEncoder(FaceEncoder):
    """InceptionResnetV1 pretrained on VGGFace2 -- the FaceNet model (Week 3)."""

    dim = 512
    name = "facenet-vggface2"

    def __init__(self) -> None:
        import torch
        from facenet_pytorch import InceptionResnetV1

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = InceptionResnetV1(pretrained="vggface2").eval().to(self._device)

    def embed(self, face_bgr: np.ndarray) -> np.ndarray | None:
        if face_bgr.size == 0:
            return None
        import cv2

        rgb = cv2.cvtColor(cv2.resize(face_bgr, (160, 160)), cv2.COLOR_BGR2RGB)
        tensor = self._torch.from_numpy(rgb).permute(2, 0, 1).float()
        tensor = (tensor - 127.5) / 128.0
        with self._torch.no_grad():
            vec = self._model(tensor.unsqueeze(0).to(self._device))[0].cpu().numpy()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else None


class FaceGallery:
    """In-memory copy of enrolled embeddings, refreshed from the database."""

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold
        self._vectors: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._people: list[tuple[int, str]] = []

    def load(self, entries: list[tuple[int, str, np.ndarray]]) -> None:
        """entries: (person_id, name, unit-length vector)"""
        if not entries:
            self._vectors = np.empty((0, 0), dtype=np.float32)
            self._people = []
            return
        self._vectors = np.vstack([e[2] for e in entries]).astype(np.float32)
        self._people = [(e[0], e[1]) for e in entries]
        log.info("face gallery loaded: %d embeddings", len(self._people))

    def match(self, vector: np.ndarray) -> Match:
        if len(self._people) == 0 or self._vectors.shape[1] != vector.shape[0]:
            return Match(person_id=None, name="unknown", distance=1.0)

        # Vectors are unit length, so cosine distance is 1 - dot product.
        distances = 1.0 - self._vectors @ vector.astype(np.float32)
        best = int(np.argmin(distances))
        dist = float(distances[best])

        if dist > self._threshold:
            return Match(person_id=None, name="unknown", distance=dist)
        pid, name = self._people[best]
        return Match(person_id=pid, name=name, distance=dist)


def build_encoder() -> FaceEncoder:
    backend = settings.recognizer_backend.lower()
    if backend == "facenet":
        log.info("recognizer: FaceNet")
        return FaceNetEncoder()
    log.info("recognizer: stub encoder")
    return StubEncoder()
