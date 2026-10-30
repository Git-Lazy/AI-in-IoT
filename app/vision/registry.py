"""Every running camera, and which one is "closest" to a device.

A device's access check is done by the camera in the same room -- the one that
can actually see who is standing at the light switch or the garage door. With no
camera in that room, the first configured camera is used.
"""

from __future__ import annotations

import logging

from app.config import settings
from app.vision.pipeline import VisionPipeline
from app.vision.recognizer import FaceRecognizer, Match

log = logging.getLogger(__name__)


class CameraRegistry:
    def __init__(self) -> None:
        self._pipelines: dict[str, VisionPipeline] = {}
        self.recognizer: FaceRecognizer | None = None

    def start(self) -> None:
        """Start one pipeline per configured camera. Failures are per camera."""
        try:
            # One face model shared by every camera: it's the heavy part.
            self.recognizer = FaceRecognizer()
        except Exception as exc:
            log.warning("face models failed to load (%s); vision disabled", exc)
            return
        for name, room, source in settings.camera_list:
            try:
                pipeline = VisionPipeline(name, room, source, self.recognizer)
                pipeline.start()
                self._pipelines[name] = pipeline
            except Exception as exc:
                # No camera attached is a normal dev state; the rest of the
                # platform stays usable so API and device work can continue.
                log.warning("camera %s not started: %s", name, exc)

    def stop(self) -> None:
        for pipeline in self._pipelines.values():
            pipeline.stop()
        self._pipelines.clear()

    @property
    def running(self) -> bool:
        return bool(self._pipelines)

    def all(self) -> list[VisionPipeline]:
        return list(self._pipelines.values())

    def get(self, name: str | None) -> VisionPipeline | None:
        if name is None:
            return self.default
        return self._pipelines.get(name)

    @property
    def default(self) -> VisionPipeline | None:
        return next(iter(self._pipelines.values()), None)

    def for_room(self, room: str) -> VisionPipeline | None:
        """The camera that should verify someone using a device in `room`."""
        for pipeline in self._pipelines.values():
            if room and pipeline.room.lower() == room.lower():
                return pipeline
        return self.default

    def room_of(self, camera_name: str) -> str:
        pipeline = self._pipelines.get(camera_name)
        return pipeline.room if pipeline else ""

    def present_anywhere(self) -> list[tuple[Match, str]]:
        """(person, camera name) for everyone trusted on any camera right now."""
        return [(m, p.camera_name) for p in self._pipelines.values() for m in p.present_people()]

    def reload_galleries(self) -> None:
        for pipeline in self._pipelines.values():
            pipeline.reload_gallery()


cameras = CameraRegistry()
