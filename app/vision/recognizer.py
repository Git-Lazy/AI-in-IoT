"""Face recognition: find a face, turn it into an embedding, match it to a person.

Three stages, each swappable:
    FaceCropper   -- find and crop the face   (MTCNN, or Haar as a fallback)
    FaceEncoder   -- crop -> unit-length vector (FaceNet, or a stub)
    FaceGallery   -- vector -> person          (nearest enrolled face + margin test)

`FaceRecognizer` bundles the first two. Enrollment and the live pipeline must both
use the *same* bundle -- a vector only means something next to vectors made by the
same cropper and encoder -- which is why the stored `model_name` names both.
"""

from __future__ import annotations

import logging
import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from app.config import settings

log = logging.getLogger(__name__)

Box = tuple[int, int, int, int]    # x, y, w, h


@dataclass
class Match:
    person_id: int | None     # None => unknown face
    name: str
    distance: float           # cosine distance; lower is a better match


# ------------------------------------------------------------------- croppers


class FaceCropper(ABC):
    name: str = "base"

    @abstractmethod
    def find(self, image_bgr: np.ndarray) -> Box | None:
        """Box of the largest usable face, or None."""

    @abstractmethod
    def crop(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """The face, cropped (and aligned, where supported), or None."""


class HaarCropper(FaceCropper):
    """OpenCV's Haar cascade. Weak but dependency-free; used with the stub encoder.

    Not thread-safe: give each thread its own instance.
    """

    name = "haar"

    def __init__(self) -> None:
        import cv2

        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)

    def find(self, image_bgr: np.ndarray) -> Box | None:
        if image_bgr.size == 0:
            return None
        import cv2

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))

    def crop(self, image_bgr: np.ndarray) -> np.ndarray | None:
        box = self.find(image_bgr)
        if box is None:
            return None
        fx, fy, fw, fh = box
        return image_bgr[fy : fy + fh, fx : fx + fw]


class MtcnnCropper(FaceCropper):
    """MTCNN detection + eye alignment -- the cropper FaceNet was trained with.

    Three things make this more precise than Haar:
      * MTCNN finds faces far more reliably, with a confidence score to gate on.
      * Its eye landmarks let us rotate the face level before cropping, so a
        tilted head doesn't look like a different person.
      * Faces that are too small or too blurry are rejected outright. A bad
        crop produces a confidently wrong embedding; no crop is better.
    """

    name = "mtcnn"
    MARGIN = 20 / (160 - 20)    # facenet-pytorch's default margin, as a box fraction

    def __init__(self) -> None:
        import torch
        from facenet_pytorch import MTCNN

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._mtcnn = MTCNN(keep_all=True, min_face_size=40, device=device)
        self._min_prob = settings.face_min_confidence
        self._min_size = settings.face_min_size
        self._min_sharpness = settings.face_min_sharpness

    def _detect(self, image_bgr: np.ndarray):
        """(box x1,y1,x2,y2, landmarks) of the largest face that passes the gates."""
        if image_bgr.size == 0:
            return None
        import cv2

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = self._mtcnn.detect(rgb, landmarks=True)
        if boxes is None:
            return None

        best = None
        for box, prob, marks in zip(boxes, probs, landmarks):
            x1, y1, x2, y2 = box
            side = min(x2 - x1, y2 - y1)
            if prob < self._min_prob or side < self._min_size:
                continue
            if best is None or side > best[2]:
                best = (box, marks, side)
        if best is None:
            return None

        box, marks, _ = best
        x1, y1, x2, y2 = (int(v) for v in box)
        region = image_bgr[max(y1, 0) : y2, max(x1, 0) : x2]
        if region.size == 0 or _sharpness(region) < self._min_sharpness:
            return None
        return box, marks

    def find(self, image_bgr: np.ndarray) -> Box | None:
        hit = self._detect(image_bgr)
        if hit is None:
            return None
        x1, y1, x2, y2 = (int(v) for v in hit[0])
        return x1, y1, x2 - x1, y2 - y1

    def crop(self, image_bgr: np.ndarray) -> np.ndarray | None:
        hit = self._detect(image_bgr)
        if hit is None:
            return None
        import cv2

        (x1, y1, x2, y2), marks = hit
        left_eye, right_eye = marks[0], marks[1]

        # Rotate the whole frame about the eye midpoint so the eyes are level.
        angle = math.degrees(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        eyes = (float((left_eye[0] + right_eye[0]) / 2), float((left_eye[1] + right_eye[1]) / 2))
        rot = cv2.getRotationMatrix2D(eyes, angle, 1.0)
        h, w = image_bgr.shape[:2]
        aligned = cv2.warpAffine(image_bgr, rot, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Same box, moved with the rotation, plus FaceNet's standard margin.
        cx, cy = rot @ np.array([(x1 + x2) / 2, (y1 + y2) / 2, 1.0])
        half_w = (x2 - x1) * (1 + self.MARGIN) / 2
        half_h = (y2 - y1) * (1 + self.MARGIN) / 2
        top, bottom = int(max(cy - half_h, 0)), int(min(cy + half_h, h))
        left, right = int(max(cx - half_w, 0)), int(min(cx + half_w, w))
        face = aligned[top:bottom, left:right]
        return face if face.size else None


def _sharpness(image_bgr: np.ndarray) -> float:
    """Variance of the Laplacian: low means blurry (motion, focus, or tiny)."""
    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def build_cropper() -> FaceCropper:
    if settings.face_detector.lower() == "mtcnn":
        try:
            return MtcnnCropper()
        except ImportError:
            log.warning("facenet-pytorch not installed; falling back to Haar face detection")
    return HaarCropper()


# ------------------------------------------------------------------- encoders


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
    """InceptionResnetV1 pretrained on VGGFace2 -- the FaceNet model."""

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


def build_encoder() -> FaceEncoder:
    backend = settings.recognizer_backend.lower()
    if backend == "facenet":
        log.info("recognizer: FaceNet")
        return FaceNetEncoder()
    log.info("recognizer: stub encoder")
    return StubEncoder()


# ------------------------------------------------------------------ the bundle


class FaceRecognizer:
    """Cropper + encoder, shared safely between the vision thread and the API."""

    def __init__(self, cropper: FaceCropper | None = None, encoder: FaceEncoder | None = None):
        self.cropper = cropper or build_cropper()
        self.encoder = encoder or build_encoder()
        self._lock = threading.Lock()
        log.info("face pipeline: %s", self.model_name)

    @property
    def model_name(self) -> str:
        """Stored with every embedding; only same-named vectors are compared."""
        return f"{self.encoder.name}/{self.cropper.name}"

    def find(self, image_bgr: np.ndarray) -> Box | None:
        with self._lock:
            return self.cropper.find(image_bgr)

    def embed_image(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """Find, crop, and embed the largest face in an image."""
        with self._lock:
            face = self.cropper.crop(image_bgr)
            if face is None:
                return None
            return self.encoder.embed(face)


# -------------------------------------------------------------------- matching


class FaceGallery:
    """In-memory copy of enrolled embeddings, refreshed from the database.

    A face is accepted as a person only if it is (a) close enough to one of their
    enrolled faces and (b) clearly closer to them than to anyone else. The margin
    test is what stops two similar-looking people being confused for each other.
    """

    def __init__(self, threshold: float, margin: float = 0.0) -> None:
        self._threshold = threshold
        self._margin = margin
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

        # Closest enrolled face per person.
        best_per_person: dict[int, tuple[float, str]] = {}
        for (pid, name), dist in zip(self._people, distances):
            if pid not in best_per_person or dist < best_per_person[pid][0]:
                best_per_person[pid] = (float(dist), name)
        ranked = sorted(best_per_person.items(), key=lambda kv: kv[1][0])

        pid, (dist, name) = ranked[0]
        if dist > self._threshold:
            return Match(person_id=None, name="unknown", distance=dist)
        if len(ranked) > 1 and ranked[1][1][0] - dist < self._margin:
            return Match(person_id=None, name="unknown", distance=dist)   # ambiguous
        return Match(person_id=pid, name=name, distance=dist)
