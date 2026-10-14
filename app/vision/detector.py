"""Object/person detection.

`Detector` is the contract the pipeline depends on. `StubDetector` satisfies it
today so the whole system runs without torch installed; `YoloDetector` is the
Week-3 drop-in. Nothing downstream changes when you switch.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from app.config import settings

log = logging.getLogger(__name__)


@dataclass
class Detection:
    label: str                     # "person", "dog", "car", ...
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    track_id: int | None = None     # stable across frames when tracking is on

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) // 2, (y1 + y2) // 2


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detections for one BGR frame."""


class StubDetector(Detector):
    """Reports nothing. Keeps the pipeline honest until YOLO is wired in."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return []


class MotionDetector(Detector):
    """Classic background subtraction -- a real, dependency-free baseline.

    Useful as a control to measure YOLO against: how many of these blobs are
    actually people? That comparison is a good source of evaluation numbers for
    the Week-10 writeup.
    """

    def __init__(self, min_area: int = 1500) -> None:
        import cv2

        self._bg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self._min_area = min_area

    def detect(self, frame: np.ndarray) -> list[Detection]:
        import cv2

        mask = self._bg.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        out: list[Detection] = []
        for c in contours:
            if cv2.contourArea(c) < self._min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            out.append(Detection(label="motion", confidence=0.5, box=(x, y, x + w, y + h)))
        return out


class YoloDetector(Detector):
    """Ultralytics YOLO11 with built-in tracking (Week 3).

    `model.track` gives persistent IDs across frames, which is what lets the
    rules engine tell "one person lingering" from "six separate detections".
    """

    def __init__(self, model_path: str, confidence: float) -> None:
        from ultralytics import YOLO  # imported lazily: heavy dependency

        self._model = YOLO(model_path)
        self._conf = confidence

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.track(
            frame, persist=True, conf=self._conf, verbose=False, classes=[0]  # 0 = person
        )
        out: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                out.append(
                    Detection(
                        label=self._model.names[int(b.cls[0])],
                        confidence=float(b.conf[0]),
                        box=(x1, y1, x2, y2),
                        track_id=int(b.id[0]) if b.id is not None else None,
                    )
                )
        return out


def build_detector() -> Detector:
    backend = settings.detector_backend.lower()
    if backend == "yolo":
        log.info("detector: YOLO (%s)", settings.yolo_model_path)
        return YoloDetector(settings.yolo_model_path, settings.detection_confidence)
    if backend == "motion":
        log.info("detector: motion baseline")
        return MotionDetector()
    log.info("detector: stub (no detections)")
    return StubDetector()
