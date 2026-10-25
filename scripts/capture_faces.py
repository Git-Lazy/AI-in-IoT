"""Capture enrollment photos from the same camera the backend watches.

    python scripts/capture_faces.py --name "Samuel"

Opens a live preview on CAMERA_SOURCE (or --source). A green box means the face
detector sees you; press SPACE to save that frame. Frames without a detectable
face are refused, because enroll_face.py would just skip them anyway.

Keys:  SPACE = capture   Q / ESC = quit

Then enroll what you captured:

    python scripts/enroll_face.py --name "Samuel" --images data/faces/samuel
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from app.config import settings  # noqa: E402
from app.vision.recognizer import FaceCropper  # noqa: E402

WINDOW = "capture faces  |  SPACE = capture   Q = quit"


def parse_source(raw: str | None) -> int | str:
    if raw is None:
        return settings.camera_index_or_url
    return int(raw) if raw.isdigit() else raw


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture face photos for enrollment")
    parser.add_argument("--name", required=True, help="person's name, e.g. Samuel")
    parser.add_argument("--source", help="camera index or URL (default: CAMERA_SOURCE)")
    parser.add_argument("--out", type=Path, help="output folder (default: data/faces/<name>)")
    parser.add_argument("--count", type=int, default=10, help="target number of photos")
    args = parser.parse_args()

    out = args.out or Path("data/faces") / args.name.lower().replace(" ", "_")
    out.mkdir(parents=True, exist_ok=True)
    existing = len(list(out.glob("*.jpg")))

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"could not open camera source {source!r} -- try --source 1")
        return 1
    print(f"camera {source!r} open; saving to {out} ({existing} photo(s) already there)")

    # Same detector enroll_face.py uses: if it finds a face here, enrollment will too.
    cropper = FaceCropper()
    saved = 0
    flash_until = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera stopped returning frames")
                return 1

            box = cropper.find(frame)
            view = frame.copy()
            if box is not None:
                x, y, w, h = box
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 200, 0), 2)
                status, color = "face found - SPACE to capture", (0, 200, 0)
            else:
                status, color = "no face - face the camera, more light", (0, 0, 220)

            if time.time() < flash_until:
                cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), (255, 255, 255), 12)

            cv2.putText(view, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(
                view, f"captured {saved}/{args.count}", (10, view.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if box is None:
                    print("  no face in frame -- not saved")
                    continue
                path = out / f"{int(time.time() * 1000)}.jpg"
                cv2.imwrite(str(path), frame)    # full frame; enrollment does the crop
                saved += 1
                flash_until = time.time() + 0.15
                print(f"  + {path.name}  ({saved}/{args.count})")
                if saved >= args.count:
                    print("target reached")
                    break
            # Closing the window with the X button.
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nsaved {saved} photo(s) to {out}")
    if saved:
        print(f'next: python scripts/enroll_face.py --name "{args.name}" --images {out.as_posix()}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
