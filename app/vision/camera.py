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

# A live camera that hasn't produced a *new* picture for this long is stuck
# (unplugged, grabbed by another app, driver hiccup). Real sensors are noisy, so
# even a still room never gives two pixel-identical frames.
STALL_SECONDS = 5.0
RETRY_SECONDS = 2.0


def _is_live(source: int | str) -> bool:
    return isinstance(source, int) or str(source).lower().startswith(("rtsp:", "http:", "https:"))


def _signature(frame: np.ndarray) -> int:
    return hash(frame[::8, ::8].tobytes())


class Camera:
    """Reads frames in a background thread and keeps only the latest one.

    Dropping stale frames is deliberate: if inference is slower than the camera,
    we want the *current* state of the room, not a growing backlog.

    A live camera that stops delivering new frames is reopened until it comes
    back; meanwhile read() returns None and `stalled` is True, so the dashboard
    shows "no signal" instead of a frozen picture.
    """

    def __init__(self, source: int | str, fps: int = 10, name: str = "camera") -> None:
        self.source = source
        self.name = name
        self.stalled = False
        self._live = _is_live(source)
        self._interval = 1.0 / max(fps, 1)
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap = None
        self._last_sig: int | None = None
        self._last_new = 0.0

    def _open(self):
        import cv2

        from app.config import settings

        api = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}.get(settings.camera_api.lower())
        if api is not None and isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source, api)
        else:
            cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self) -> None:
        self._cap = self._open()
        if self._cap is None:
            raise RuntimeError(f"could not open camera source: {self.source!r}")
        self._last_new = time.monotonic()
        self._thread = threading.Thread(target=self._loop, name=f"camera:{self.name}", daemon=True)
        self._thread.start()
        log.info("camera %s started on source %r", self.name, self.source)

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read() if self._cap is not None else (False, None)

            if not self._live:
                if not ok:
                    # Video file ended -- loop it so test footage runs continuously.
                    import cv2

                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    if not self._cap.read()[0]:
                        log.warning("camera %s lost its source", self.name)
                        self._stop.wait(1.0)
                    continue
                self._store(frame)
                self._stop.wait(self._interval)
                continue

            now = time.monotonic()
            if ok:
                sig = _signature(frame)
                if sig != self._last_sig:
                    self._last_sig, self._last_new = sig, now
                    if self.stalled:
                        self.stalled = False
                        log.info("camera %s is live again", self.name)
                if now - self._last_new < STALL_SECONDS:
                    self._store(frame)
                    self._stop.wait(self._interval)
                    continue
            elif now - self._last_new < STALL_SECONDS:
                self._stop.wait(0.1)  # a dropped frame or two is normal
                continue

            self._reconnect()

    def _store(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def _reconnect(self) -> None:
        if not self.stalled:
            log.warning(
                "camera %s: no new picture for %.0fs -- reopening source %r "
                "(unplugged, or another app using it?)", self.name, STALL_SECONDS, self.source,
            )
        self.stalled = True
        with self._lock:
            self._frame = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._stop.wait(RETRY_SECONDS):
            return
        self._cap = self._open()
        self._last_sig, self._last_new = None, time.monotonic()

    def read(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._cap is not None:
            self._cap.release()
        log.info("camera %s stopped", self.name)
