from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2

from weights_config import resolve_yolo_weights, yolo_class_filter


@dataclass(frozen=True)
class Detection:
    bbox: list[float]
    conf: float


@dataclass(frozen=True)
class FrameDetections:
    frame: int
    detections: list[Detection]


class PersonDetector:
    """YOLOv8 person detector. Supports COCO or soccer fine-tuned weights."""

    def __init__(
        self,
        weights: str | None = None,
        conf: float = 0.1,
        iou: float = 0.5,
        classes: list[int] | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.weights = resolve_yolo_weights(weights)
        self.conf = conf
        self.iou = iou
        self.classes = classes if classes is not None else yolo_class_filter()
        self.model = YOLO(self.weights)

    def detect_frame(self, frame) -> list[Detection]:
        kwargs: dict = {
            "conf": self.conf,
            "iou": self.iou,
            "verbose": False,
        }
        if self.classes is not None:
            kwargs["classes"] = self.classes

        result = self.model.predict(frame, **kwargs)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        return [
            Detection(bbox=[float(x) for x in box], conf=float(score))
            for box, score in zip(boxes, scores)
        ]

    def detect_all(self, frames_dir: str | Path) -> list[FrameDetections]:
        frames_dir = Path(frames_dir)
        frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
        results: list[FrameDetections] = []

        for frame_path in frame_paths:
            frame_idx = int(frame_path.stem.split("_")[1])
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise ValueError(f"Could not read frame: {frame_path}")
            results.append(
                FrameDetections(frame=frame_idx, detections=self.detect_frame(frame))
            )

        return results
