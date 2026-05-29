from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from weights_config import resolve_ball_weights


@dataclass(frozen=True)
class BallDetection:
    bbox: list[float]
    conf: float


class BallDetector:
    """Optional YOLOv8 ball detector (sandbox only)."""

    def __init__(
        self,
        weights: str | None = None,
        conf: float = 0.25,
        iou: float = 0.5,
    ) -> None:
        from ultralytics import YOLO

        resolved = resolve_ball_weights(weights)
        if resolved is None:
            raise FileNotFoundError(
                "Ball weights not found. Place yolov8n_ball.pt in weights/ "
                "or set BALL_WEIGHTS."
            )
        self.weights = resolved
        self.conf = conf
        self.iou = iou
        self.model = YOLO(self.weights)

    def detect_frame(self, frame) -> list[BallDetection]:
        result = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        return [
            BallDetection(bbox=[float(x) for x in box], conf=float(score))
            for box, score in zip(boxes, scores)
        ]

    @staticmethod
    def try_create(
        weights: str | None = None,
        conf: float = 0.25,
        iou: float = 0.5,
    ) -> BallDetector | None:
        try:
            return BallDetector(weights=weights, conf=conf, iou=iou)
        except FileNotFoundError:
            return None
