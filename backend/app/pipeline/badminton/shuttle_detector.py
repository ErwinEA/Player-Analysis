"""YOLO shuttlecock detection for badminton rally tracking."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from backend.app.pipeline.inference_device import resolve_ultralytics_device

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]

_shuttle_yolo_model = None
_shuttle_yolo_weights: str | None = None


def _shared_shuttle_yolo(weights: str):
    """Process-wide lazy singleton for shuttle YOLO weights."""
    global _shuttle_yolo_model, _shuttle_yolo_weights
    if _shuttle_yolo_model is None or _shuttle_yolo_weights != weights:
        from ultralytics import YOLO

        _shuttle_yolo_model = YOLO(weights)
        _shuttle_yolo_weights = weights
    return _shuttle_yolo_model


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def resolve_shuttle_weights(explicit: str | None = None) -> str | None:
    if explicit and Path(explicit).is_file():
        return explicit
    env = os.environ.get("SHUTTLE_WEIGHTS", "").strip()
    if env and Path(env).is_file():
        return env
    candidate = _REPO_ROOT / "backend" / "weights" / "yolov8n_shuttle.pt"
    if candidate.is_file():
        return str(candidate)
    return None


@dataclass(frozen=True)
class ShuttleDetection:
    cx_px: float
    cy_px: float
    conf: float

    @property
    def center_px(self) -> tuple[float, float]:
        return (self.cx_px, self.cy_px)


class ShuttleDetector:
    """YOLO shuttlecock detector. Weights from SHUTTLE_WEIGHTS or backend/weights/."""

    def __init__(
        self,
        weights: str | None = None,
        conf: float | None = None,
        iou: float | None = None,
    ) -> None:
        self.conf = conf if conf is not None else _env_float("SHUTTLE_CONF", 0.25)
        self.iou = iou if iou is not None else _env_float("SHUTTLE_IOU", 0.5)

        resolved = resolve_shuttle_weights(weights)
        self.weights: str | None = resolved
        self.available = False
        self._model = None
        self._device = resolve_ultralytics_device()

        if resolved is None:
            logger.warning(
                "Shuttle weights not found — rally detection disabled. "
                "Place yolov8n_shuttle.pt in backend/weights/ or set SHUTTLE_WEIGHTS."
            )
            return

        try:
            self._model = _shared_shuttle_yolo(resolved)
            self.available = True
            logger.info("ShuttleDetector YOLO on device=%s", self._device)
        except Exception as exc:
            logger.warning("Failed to load shuttle YOLO weights %s: %s", resolved, exc)
            self.available = False

    def detect(self, frame_bgr: NDArray[np.uint8]) -> ShuttleDetection | None:
        """Highest-confidence shuttle detection in the frame, if any."""
        if not self.available or self._model is None:
            return None

        result = self._model.predict(
            frame_bgr,
            conf=self.conf,
            iou=self.iou,
            device=self._device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        best_i = int(np.argmax(scores))
        x1, y1, x2, y2 = boxes[best_i][:4]
        return ShuttleDetection(
            cx_px=float((x1 + x2) / 2.0),
            cy_px=float((y1 + y2) / 2.0),
            conf=float(scores[best_i]),
        )
