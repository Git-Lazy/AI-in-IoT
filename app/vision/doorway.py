"""Door lines: telling "walked in" apart from "is standing in view".

A camera inside the house, facing the front door, gets a line drawn across the
doorway. Every tracked person is followed by their *feet* (the bottom-center of
their box). Coming in, the feet move from the far side of the line to the near
side; that is an entry. Someone already inside who wanders into view never
crosses, so they never trigger an arrival.

Two things keep a crossing honest:
  * a dead band either side of the line -- a person standing on the threshold
    is on neither side, so they can't flicker in/out/in;
  * confirmation -- the feet must stay on the new side for several frames.

Configured with DOOR_LINES, one entry per camera, coordinates as fractions of
the frame (0..1, origin top-left):

    DOOR_LINES=front_door:0.15,0.70,0.85,0.70

"Inside" is the side of the line facing the bottom-center of the image --
right for a camera mounted inside looking at the door. Add ":flip" if your
camera sees it the other way round.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from app.config import settings

log = logging.getLogger(__name__)

INSIDE, OUTSIDE = 1, -1


@dataclass(frozen=True)
class DoorLine:
    x1: float
    y1: float
    x2: float
    y2: float
    flip: bool = False

    def offset(self, x: float, y: float) -> float:
        """Signed distance of (x, y) from the line, in frame fractions. > 0 = inside."""
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        length = math.hypot(dx, dy)
        if length == 0:
            return 0.0
        cross = dx * (y - self.y1) - dy * (x - self.x1)
        # Orient so the bottom-center of the frame is on the positive side.
        ref = dx * (1.0 - self.y1) - dy * (0.5 - self.x1)
        sign = 1.0 if ref >= 0 else -1.0
        if self.flip:
            sign = -sign
        return sign * cross / length

    def pixels(self, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
        return ((round(self.x1 * width), round(self.y1 * height)),
                (round(self.x2 * width), round(self.y2 * height)))


def parse_door_lines(spec: str) -> dict[str, DoorLine]:
    """"cam:x1,y1,x2,y2[:flip];cam2:..." -> {camera name: DoorLine}. Bad entries are skipped."""
    lines: dict[str, DoorLine] = {}
    for entry in filter(None, (e.strip() for e in spec.split(";"))):
        try:
            name, coords, *opts = (p.strip() for p in entry.split(":"))
            values = [float(v) for v in coords.split(",")]
            if len(values) != 4 or not all(0.0 <= v <= 1.0 for v in values):
                raise ValueError("need four numbers between 0 and 1")
            lines[name] = DoorLine(*values, flip="flip" in opts)
        except ValueError as exc:
            log.warning("ignoring door line %r: %s", entry, exc)
    return lines


def door_line_for(camera_name: str) -> DoorLine | None:
    return parse_door_lines(settings.door_lines).get(camera_name)


def feet(box: tuple[int, int, int, int], width: int, height: int) -> tuple[float, float]:
    """Where a person is standing: bottom-center of their box, as frame fractions."""
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2 / width, min(y2, height) / height)


class Crossing:
    """One track's side of the door line, with a dead band and confirmation."""

    def __init__(self) -> None:
        self.side = 0            # confirmed side: INSIDE, OUTSIDE, or 0 = not yet known
        self._pending = 0
        self._count = 0

    def update(self, offset: float) -> str | None:
        """Feed one frame's offset. Returns "in" or "out" when a crossing is confirmed."""
        band = settings.door_band
        observed = INSIDE if offset > band else OUTSIDE if offset < -band else 0
        if observed == 0:                    # on the threshold: no evidence either way
            return None
        if observed == self.side:
            self._pending = self._count = 0
            return None
        if observed == self._pending:
            self._count += 1
        else:
            self._pending, self._count = observed, 1
        if self._count < settings.door_confirm_frames:
            return None

        previous, self.side = self.side, observed
        self._pending = self._count = 0
        if previous == OUTSIDE:
            return "in"
        if previous == INSIDE:
            return "out"
        return None                          # first sighting just establishes the side
