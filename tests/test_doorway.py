"""Door lines: walking in is an arrival; standing in view is not."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from app.events import EventType
from app.vision.detector import Detection
from app.vision.doorway import Crossing, DoorLine, feet, parse_door_lines
from app.vision.recognizer import Match

LINE = DoorLine(0.1, 0.6, 0.9, 0.6)       # across the frame, a bit below the middle


# ------------------------------------------------------------------ geometry


def test_inside_is_the_side_toward_the_bottom_of_the_frame():
    assert LINE.offset(0.5, 0.9) > 0                     # near the camera
    assert LINE.offset(0.5, 0.3) < 0                     # out by the door
    assert DoorLine(0.9, 0.6, 0.1, 0.6).offset(0.5, 0.9) > 0   # point order doesn't matter
    assert DoorLine(0.1, 0.6, 0.9, 0.6, flip=True).offset(0.5, 0.9) < 0
    assert LINE.offset(0.5, 0.7) == pytest.approx(0.1)


def test_parse_door_lines_skips_bad_entries():
    lines = parse_door_lines("front_door:0.1,0.6,0.9,0.6; garage:0,0.5,1,0.5:flip; bad:1,2; junk")
    assert set(lines) == {"front_door", "garage"}
    assert lines["garage"].flip and not lines["front_door"].flip
    assert parse_door_lines("") == {}


def test_feet_are_bottom_center_of_the_box():
    assert feet((100, 50, 300, 400), 640, 480) == (200 / 640, 400 / 480)
    assert feet((0, 0, 10, 999), 640, 480)[1] == 1.0     # box running off the frame


# ------------------------------------------------------------------ crossing


def _walk(*offsets: float) -> list[str]:
    crossing = Crossing()
    return [r for r in (crossing.update(o) for o in offsets) if r]


def test_walking_in_is_one_entry():
    assert _walk(-0.2, -0.2, -0.2, -0.1, 0.0, 0.1, 0.2, 0.2, 0.2, 0.2) == ["in"]


def test_walking_out_is_detected_too():
    assert _walk(0.2, 0.2, 0.2, -0.1, -0.2, -0.2) == ["out"]


def test_standing_on_the_threshold_does_nothing():
    # Swaying inside the dead band, never clearly on either side.
    assert _walk(-0.2, -0.2, -0.2, *[0.02, -0.02, 0.01, -0.01] * 10) == []


def test_brief_jitter_across_the_line_is_not_a_crossing():
    # One or two frames over the line (a detector wobble), then back.
    assert _walk(-0.2, -0.2, -0.2, 0.1, 0.1, -0.2, -0.2, 0.1, -0.2) == []


def test_appearing_already_inside_is_not_an_entry():
    # Walked over from the living room: first seen inside, never crossed.
    assert _walk(0.3, 0.3, 0.3, 0.2, 0.25) == []


def test_in_out_in():
    assert _walk(-0.2, -0.2, -0.2, 0.2, 0.2, 0.2, -0.2, -0.2, -0.2, 0.2, 0.2, 0.2) == ["in", "out", "in"]


# ------------------------------------------------------------------ pipeline


@pytest.fixture
def door_pipeline(monkeypatch):
    from app.vision import pipeline as pl

    monkeypatch.setattr(settings, "door_lines", "front_door:0.1,0.6,0.9,0.6")
    sent: list = []
    clock = [1000.0]
    monkeypatch.setattr("app.vision.pipeline.bus.publish_threadsafe", sent.append)
    monkeypatch.setattr("app.vision.pipeline.time.monotonic", lambda: clock[0])
    vp = pl.VisionPipeline("front_door", "Entry")
    assert vp.door is not None
    return vp, sent, clock


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def _step(vp, clock, feet_y: float, track_id: int = 4) -> None:
    """One frame with a person whose feet are at feet_y (fraction of height)."""
    y2 = int(feet_y * 480)
    vp._handle(FRAME, [Detection("person", 0.9, (280, max(y2 - 300, 0), 360, y2), track_id)])
    clock[0] += 0.1


def _entries(sent):
    return [e for e in sent if e.type == EventType.PERSON_ENTERED]


def test_known_person_walking_in_is_announced_with_their_name(door_pipeline):
    from app.vision.pipeline import TrackState

    vp, sent, clock = door_pipeline
    vp._tracks[4] = TrackState(identity=Match(person_id=1, name="Samuel", distance=0.2))
    for y in (0.4, 0.4, 0.4, 0.5, 0.7, 0.75, 0.8, 0.85):
        _step(vp, clock, y)

    [entry] = _entries(sent)
    assert entry.person_id == 1
    assert entry.payload["name"] == "Samuel" and entry.payload["camera"] == "front_door"
    for _ in range(50):                                  # standing inside afterwards
        _step(vp, clock, 0.85)
    assert len(_entries(sent)) == 1


def test_entry_waits_for_the_face_then_gives_up(door_pipeline):
    vp, sent, clock = door_pipeline
    for y in (0.4, 0.4, 0.4, 0.7, 0.75, 0.8):
        _step(vp, clock, y)
    assert _entries(sent) == []                          # crossed, still hoping for a face

    clock[0] += settings.door_identify_seconds
    _step(vp, clock, 0.8)
    [entry] = _entries(sent)
    assert entry.person_id is None and entry.payload["face"] == "none"


def test_entry_is_announced_when_the_track_disappears(door_pipeline):
    from app.vision import pipeline as pl

    vp, sent, clock = door_pipeline
    for y in (0.4, 0.4, 0.4, 0.7, 0.75, 0.8):
        _step(vp, clock, y)
    clock[0] += pl.TRACK_TTL_SECONDS + 1
    vp._handle(FRAME, [])                                # walked out of view
    assert len(_entries(sent)) == 1 and 4 not in vp._tracks


def test_camera_without_door_line_never_announces_entries(monkeypatch):
    from app.vision import pipeline as pl

    monkeypatch.setattr(settings, "door_lines", "front_door:0.1,0.6,0.9,0.6")
    sent: list = []
    monkeypatch.setattr("app.vision.pipeline.bus.publish_threadsafe", sent.append)
    vp = pl.VisionPipeline("living_room", "Living Room")
    assert vp.door is None
    for y in (0.4, 0.4, 0.4, 0.7, 0.8, 0.9):
        vp._handle(FRAME, [Detection("person", 0.9, (280, 0, 360, int(y * 480)), 4)])
    assert _entries(sent) == []
