"""A live camera that stops producing new pictures is noticed and reopened."""

import time

import numpy as np

from app.vision import camera as cam_mod
from app.vision.camera import Camera


class FakeCap:
    """Live frames until `freeze`, then the same frame forever (a stuck camera)."""

    def __init__(self, freeze_after=None, fail=False):
        self.n, self.freeze_after, self.fail, self.released = 0, freeze_after, fail, False

    def read(self):
        if self.fail:
            return False, None
        self.n += 1
        if self.freeze_after is not None and self.n > self.freeze_after:
            return True, np.zeros((48, 64, 3), np.uint8)
        return True, np.random.randint(0, 255, (48, 64, 3), np.uint8)

    def release(self):
        self.released = True


def _wait(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def _camera(monkeypatch, caps):
    monkeypatch.setattr(cam_mod, "STALL_SECONDS", 0.3)
    monkeypatch.setattr(cam_mod, "RETRY_SECONDS", 0.05)
    opened = []

    def fake_open(self):
        cap = caps.pop(0) if caps else FakeCap()
        opened.append(cap)
        return cap

    monkeypatch.setattr(Camera, "_open", fake_open)
    return Camera(0, fps=50, name="test"), opened


def test_frozen_camera_shows_no_frame_then_recovers(monkeypatch):
    c, opened = _camera(monkeypatch, [FakeCap(freeze_after=3), FakeCap(fail=True)])
    c.start()
    try:
        assert _wait(lambda: c.read() is not None)
        assert _wait(lambda: c.stalled)          # same picture for too long
        assert c.read() is None                  # no frozen frame served
        assert opened[0].released
        assert _wait(lambda: not c.stalled)      # third open is a healthy camera
        assert c.read() is not None
    finally:
        c.stop()


def test_healthy_camera_never_stalls(monkeypatch):
    c, _ = _camera(monkeypatch, [FakeCap()])
    c.start()
    try:
        time.sleep(0.6)
        assert not c.stalled and c.read() is not None
    finally:
        c.stop()


def test_video_file_is_not_treated_as_live():
    assert not cam_mod._is_live("clips/walk_in.mp4")
    assert cam_mod._is_live(1) and cam_mod._is_live("rtsp://cam/stream")
