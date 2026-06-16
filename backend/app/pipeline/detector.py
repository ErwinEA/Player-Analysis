from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING

import numpy as np

from backend.app.pipeline.inference_device import resolve_ultralytics_device

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

BBoxScore = tuple[list[float], float]


class PersonDetector:
    """YOLOv8 person detector. Weights path from YOLO_WEIGHTS (default yolov8n.pt)."""

    def __init__(
        self,
        weights: str | None = None,
        conf: float | None = None,
        iou: float = 0.5,
    ) -> None:
        from ultralytics import YOLO

        self.weights = weights or os.environ.get("YOLO_WEIGHTS", "yolov8n.pt")
        self.conf = conf if conf is not None else float(os.environ.get("DET_CONF", "0.1"))
        track_low = float(os.environ.get("TRACK_LOW_THRESH", "0.1"))
        if self.conf > track_low:
            warnings.warn(
                f"DET_CONF={self.conf} is above TRACK_LOW_THRESH={track_low}; "
                "ByteTrack second-stage association needs low-confidence detections. "
                f"Lower DET_CONF to {track_low} or below.",
                stacklevel=2,
            )
        self.iou = iou
        self.device = resolve_ultralytics_device()
        raw_imgsz = os.environ.get("YOLO_IMGSZ", "").strip()
        self.imgsz = int(raw_imgsz) if raw_imgsz.isdigit() and int(raw_imgsz) > 0 else None
        self.model = YOLO(self.weights)
        logger.info("PersonDetector YOLO on device=%s", self.device)

    def __call__(self, frame: NDArray[np.uint8]) -> list[BBoxScore]:
        kwargs: dict = {
            "classes": [0],
            "conf": self.conf,
            "iou": self.iou,
            "device": self.device,
            "verbose": False,
        }
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz
        result = self.model.predict(frame, **kwargs)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        return [([float(x) for x in box], float(score)) for box, score in zip(boxes, scores)]


_person_detector: PersonDetector | None = None


def get_person_detector() -> PersonDetector:
    """Process-wide lazy singleton (avoids reloading YOLO weights per analyze request)."""
    global _person_detector
    if _person_detector is None:
        _person_detector = PersonDetector()
    return _person_detector
