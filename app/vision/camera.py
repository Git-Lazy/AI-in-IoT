"""Frame source: webcam, RTSP stream, or a video file for repeatable testing.

A video file as CAMERA_SOURCE is the cheap way to get a deterministic test
fixture -- same frames every run, which matters for measuring false positives.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

log = logging.getLogger(__name__)


class Camera:
    """Reads frames in a background thread and keeps only the latest one.

    Dropping stale frames is deliberate: if inference is slower than the camera,
    we want the *current* state of the room, not a growing backlog.
    """

    def __init__(self, source: int | str, fps: int = 10, name: str = "camera") -> None:
        self.source = source
        self.name = name
        self._interval = 1.0 / max(fps, 1)
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap = None

    def start(self) -> None:
        import cv2

        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open camera source: {self.source!r}")
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._thread = threading.Thread(target=self._loop, name=f"camera:{self.name}", daemon=True)
        self._thread.start()
        log.info("camera %s started on source %r", self.name, self.source)

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                # Video file ended -- loop it so test footage runs continuously.
                import cv2

                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if not self._cap.read()[0]:
                    log.warning("camera %s lost its source", self.name)
                    time.sleep(1.0)
                continue
            with self._lock:
                self._frame = frame
            time.sleep(self._interval)

    def read(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        log.info("camera %s stopped", self.name)
