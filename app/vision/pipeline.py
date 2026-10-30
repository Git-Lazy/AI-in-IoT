"""The vision pipeline: capture -> detect -> recognize -> publish events.

Runs on its own thread (OpenCV and torch both block) and pushes results onto the
async event bus. This is the module that turns pixels into the structured events
the rest of the platform reasons about.

Identity is decided per *track* (one person, followed across frames by YOLO), by
vote: a face has to match the same person in several recent frames before the
track is trusted as that person. One lucky -- or unlucky -- frame can't unlock a
door or raise an intruder alert.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from app.config import settings
from app.db import session_scope
from app.db.models import FaceEmbedding, Person
from app.events import Event, EventType, bus
from app.vision.camera import Camera
from app.vision.detector import Detection, build_detector
from app.vision.recognizer import FaceGallery, FaceRecognizer, Match

log = logging.getLogger(__name__)

# How often to re-confirm a recognized person who is still in frame. Must stay
# well under rules.PRESENCE_TIMEOUT or they get marked "left" mid-visit.
SEEN_HEARTBEAT_SECONDS = 30
# Once a track is trusted, re-check the face this often rather than every frame.
REVERIFY_SECONDS = 1.0
# A track nobody has seen for this long is forgotten.
TRACK_TTL_SECONDS = 5.0
# "At the camera right now" means seen within this window (used for access checks).
PRESENT_SECONDS = 2.0
# Tracker IDs can be missing (e.g. the motion backend); those share one slot.
UNTRACKED = -1


@dataclass
class TrackState:
    votes: deque = field(default_factory=lambda: deque(maxlen=settings.face_vote_window))
    identity: Match | None = None     # set once enough votes agree
    last_seen: float = 0.0
    last_embed: float = 0.0


class VisionPipeline:
    def __init__(
        self,
        camera_name: str = "front_door",
        room: str = "",
        source: int | str | None = None,
        recognizer: FaceRecognizer | None = None,
    ) -> None:
        self.camera_name = camera_name
        self.room = room
        source = settings.camera_index_or_url if source is None else source
        self._camera = Camera(source, settings.camera_fps, camera_name)
        self._detector = build_detector()
        self.recognizer = recognizer or FaceRecognizer()
        self._gallery = FaceGallery(settings.face_match_threshold, settings.face_match_margin)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        # Latest annotated frame, for the dashboard's MJPEG preview.
        self._preview: np.ndarray | None = None
        self._preview_lock = threading.Lock()

        # track_id -> what we know about that person. Written by the vision
        # thread, read by API threads (access checks), hence the lock.
        self._tracks: dict[int, TrackState] = {}
        self._tracks_lock = threading.Lock()
        # person_id -> monotonic time of the last PERSON_RECOGNIZED/PERSON_SEEN.
        self._last_seen_sent: dict[int, float] = {}
        self._last_untracked_event = 0.0

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self.reload_gallery()
        self._camera.start()
        self._thread = threading.Thread(
            target=self._run, name=f"vision:{self.camera_name}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._camera.stop()

    def reload_gallery(self) -> None:
        """Pull enrolled embeddings out of the database into memory.

        Only embeddings from the active cropper + encoder are loaded -- vectors
        from a different model live in a different space and can't be compared.
        """
        with session_scope() as session:
            rows = (
                session.query(FaceEmbedding, Person)
                .join(Person, FaceEmbedding.person_id == Person.id)
                .filter(FaceEmbedding.model_name == self.recognizer.model_name)
                .all()
            )
            entries = [
                (p.id, p.name, np.frombuffer(f.vector, dtype=np.float32))
                for f, p in rows
            ]
        self._gallery.load(entries)

    # ------------------------------------------------------------------- frames

    def latest_frame(self) -> np.ndarray | None:
        """Current raw camera frame (no overlays) -- used for enrollment capture."""
        return self._camera.read()

    def latest_preview(self) -> np.ndarray | None:
        with self._preview_lock:
            return None if self._preview is None else self._preview.copy()

    def present_people(self) -> list[Match]:
        """Trusted, known people in front of this camera right now."""
        now = time.monotonic()
        with self._tracks_lock:
            return [
                t.identity for t in self._tracks.values()
                if t.identity is not None
                and t.identity.person_id is not None
                and now - t.last_seen <= PRESENT_SECONDS
            ]

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
        now = time.monotonic()

        for det in detections:
            if det.label not in ("person", "motion"):
                continue
            key = det.track_id if det.track_id is not None else UNTRACKED

            with self._tracks_lock:
                track = self._tracks.get(key)
                is_new = track is None
                if is_new:
                    track = self._tracks[key] = TrackState()
                track.last_seen = now

            # One "person detected" per appearance, not one per frame.
            if is_new or (key == UNTRACKED and now - self._last_untracked_event > 1.0):
                if key == UNTRACKED:
                    self._last_untracked_event = now
                bus.publish_threadsafe(
                    Event(
                        type=EventType.PERSON_DETECTED,
                        source=source,
                        confidence=det.confidence,
                        payload={"box": list(det.box), "track_id": det.track_id},
                    )
                )

            self._heartbeat(det, source)

            # A trusted track only needs an occasional re-check.
            trusted = track.identity is not None and track.identity.person_id is not None
            if trusted and now - track.last_embed < REVERIFY_SECONDS:
                continue

            match = self._identify(frame, det)
            if match is None:
                continue
            track.last_embed = now
            self._vote(track, match, det, source)

        self._prune(now)

    def _vote(self, track: TrackState, match: Match, det: Detection, source: str) -> None:
        """Add one frame's verdict; announce when the track's identity settles."""
        track.votes.append(match)
        counts = Counter(m.person_id for m in track.votes)
        winner, n = counts.most_common(1)[0]
        if n < settings.face_votes_required:
            return

        # Best (closest) of the agreeing frames speaks for the track.
        decided = min((m for m in track.votes if m.person_id == winner), key=lambda m: m.distance)
        previous = track.identity
        with self._tracks_lock:
            track.identity = decided
        if previous is not None and previous.person_id == decided.person_id:
            return

        known = decided.person_id is not None
        log.info(
            "track #%s identified as %s (match %.2f, %d/%d frames agree)",
            det.track_id, decided.name, 1.0 - decided.distance, n, len(track.votes),
        )
        if known:
            self._last_seen_sent[decided.person_id] = time.monotonic()
        bus.publish_threadsafe(
            Event(
                type=EventType.PERSON_RECOGNIZED if known else EventType.UNKNOWN_PERSON,
                source=source,
                person_id=decided.person_id,
                confidence=round(1.0 - decided.distance, 3),
                payload={
                    "name": decided.name,
                    "distance": round(decided.distance, 3),
                    "track_id": det.track_id,
                    "camera": self.camera_name,
                },
            )
        )

    def _prune(self, now: float) -> None:
        with self._tracks_lock:
            stale = [k for k, t in self._tracks.items() if now - t.last_seen > TRACK_TTL_SECONDS]
            for k in stale:
                del self._tracks[k]

    def _heartbeat(self, det: Detection, source: str) -> None:
        """Keep presence alive while a recognized track stays in frame.

        Recognition is announced once per track, so without this `last_seen`
        would freeze at the first sighting and expire_presence would mark the
        person as gone while they're still standing there. Keyed on the track,
        not the face, so it holds even when they turn away from the camera.
        """
        if det.track_id is None:
            return
        track = self._tracks.get(det.track_id)
        match = track.identity if track is not None else None
        if match is None or match.person_id is None:
            return

        now = time.monotonic()
        last = self._last_seen_sent.get(match.person_id)
        if last is not None and now - last < SEEN_HEARTBEAT_SECONDS:
            return
        self._last_seen_sent[match.person_id] = now

        bus.publish_threadsafe(
            Event(
                type=EventType.PERSON_SEEN,
                source=source,
                person_id=match.person_id,
                payload={"name": match.name, "track_id": det.track_id, "camera": self.camera_name},
            )
        )

    def _identify(self, frame: np.ndarray, det: Detection) -> Match | None:
        """Find the face inside the person box and match it."""
        x1, y1, x2, y2 = det.box
        h, w = frame.shape[:2]
        region = frame[max(y1, 0) : min(y2, h), max(x1, 0) : min(x2, w)]
        vector = self.recognizer.embed_image(region)
        if vector is None:
            return None
        return self._gallery.match(vector)

    def _draw_preview(self, frame: np.ndarray, detections: list[Detection]) -> None:
        import cv2

        annotated = frame.copy()
        need = settings.face_votes_required
        for det in detections:
            x1, y1, x2, y2 = det.box
            key = det.track_id if det.track_id is not None else UNTRACKED
            track = self._tracks.get(key)
            match = track.identity if track is not None else None

            # Green = recognized, red = unknown face, gray = still verifying.
            if match is None:
                votes = len(track.votes) if track is not None else 0
                color = (160, 160, 160)
                label = f"verifying {min(votes, need)}/{need}" if votes else f"{det.label} {det.confidence:.2f}"
            elif match.person_id is not None:
                color, label = (0, 200, 0), f"{match.name}  match {1 - match.distance:.2f}"
            else:
                color, label = (0, 0, 220), f"unknown  match {1 - match.distance:.2f}"
            if det.track_id is not None:
                label += f" #{det.track_id}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated, label, (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )
        with self._preview_lock:
            self._preview = annotated
