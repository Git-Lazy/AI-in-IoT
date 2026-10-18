"""The vision pipeline: capture -> detect -> recognize -> publish events.

Runs on its own thread (OpenCV and torch both block) and pushes results onto the
async event bus. This is the module that turns pixels into the structured events
the rest of the platform reasons about.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from app.config import settings
from app.db import session_scope
from app.db.models import FaceEmbedding, Person
from app.events import Event, EventType, bus
from app.vision.camera import Camera
from app.vision.detector import Detection, build_detector
from app.vision.recognizer import FaceCropper, FaceGallery, build_encoder

log = logging.getLogger(__name__)


class VisionPipeline:
    def __init__(self, camera_name: str = "front_door") -> None:
        self.camera_name = camera_name
        self._camera = Camera(settings.camera_index_or_url, settings.camera_fps, camera_name)
        self._detector = build_detector()
        self._encoder = build_encoder()
        self._gallery = FaceGallery(settings.face_match_threshold)
        self._face_cropper = FaceCropper()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        # Latest annotated frame, for the dashboard's MJPEG preview.
        self._preview: np.ndarray | None = None
        self._preview_lock = threading.Lock()

        # track_id -> last reported person, so we announce each person once per
        # appearance instead of once per frame. Crude but effective de-duplication;
        # the Week 6-7 rules work replaces this with proper dwell logic.
        self._announced: dict[int, str] = {}

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self.reload_gallery()
        self._camera.start()
        self._thread = threading.Thread(target=self._run, name="vision", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._camera.stop()

    def reload_gallery(self) -> None:
        """Pull enrolled embeddings out of the database into memory."""
        with session_scope() as session:
            rows = (
                session.query(FaceEmbedding, Person)
                .join(Person, FaceEmbedding.person_id == Person.id)
                .all()
            )
            entries = [
                (p.id, p.name, np.frombuffer(f.vector, dtype=np.float32))
                for f, p in rows
            ]
        self._gallery.load(entries)

    # ------------------------------------------------------------------- frames

    def latest_preview(self) -> np.ndarray | None:
        with self._preview_lock:
            return None if self._preview is None else self._preview.copy()

    # --------------------------------------------------------------- main loop

    def _run(self) -> None:
        interval = 1.0 / max(settings.camera_fps, 1)
        log.info("vision pipeline running on camera %s", self.camera_name)

        while not self._stop.is_set():
            frame = self._camera.read()
            if frame is None:
                time.sleep(interval)
                continue

            try:
                detections = self._detector.detect(frame)
                self._handle(frame, detections)
                self._draw_preview(frame, detections)
            except Exception:
                log.exception("vision frame failed")

            time.sleep(interval)

    def _handle(self, frame: np.ndarray, detections: list[Detection]) -> None:
        source = f"vision:{self.camera_name}"

        for det in detections:
            if det.label not in ("person", "motion"):
                continue

            bus.publish_threadsafe(
                Event(
                    type=EventType.PERSON_DETECTED,
                    source=source,
                    confidence=det.confidence,
                    payload={"box": list(det.box), "track_id": det.track_id},
                )
            )

            match = self._identify(frame, det)
            if match is None:
                continue

            # Only announce an identity change for this track.
            if det.track_id is not None and self._announced.get(det.track_id) == match.name:
                continue
            if det.track_id is not None:
                self._announced[det.track_id] = match.name

            known = match.person_id is not None
            bus.publish_threadsafe(
                Event(
                    type=EventType.PERSON_RECOGNIZED if known else EventType.UNKNOWN_PERSON,
                    source=source,
                    person_id=match.person_id,
                    confidence=round(1.0 - match.distance, 3),
                    payload={
                        "name": match.name,
                        "distance": round(match.distance, 3),
                        "track_id": det.track_id,
                        "camera": self.camera_name,
                    },
                )
            )

    def _identify(self, frame: np.ndarray, det: Detection):
        """Crop the face region inside a person box and match it."""
        face = self._crop_face(frame, det)
        if face is None:
            return None
        vector = self._encoder.embed(face)
        if vector is None:
            return None
        return self._gallery.match(vector)

    def _crop_face(self, frame: np.ndarray, det: Detection) -> np.ndarray | None:
        """Find the face inside the person box (same cropper enrollment uses)."""
        x1, y1, x2, y2 = det.box
        h, w = frame.shape[:2]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        return self._face_cropper.crop(frame[y1:y2, x1:x2])

    def _draw_preview(self, frame: np.ndarray, detections: list[Detection]) -> None:
        import cv2

        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"{det.label} {det.confidence:.2f}"
            if det.track_id is not None:
                label += f" #{det.track_id}"
            cv2.putText(
                annotated, label, (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1,
            )
        with self._preview_lock:
            self._preview = annotated
