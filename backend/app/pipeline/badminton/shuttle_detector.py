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


def shuttle_health() -> dict[str, object]:
    """Health payload for /health (weights on disk; model loads on first badminton analyze)."""
    weights = resolve_shuttle_weights()
    if weights is None:
        return {
            "weights_found": False,
            "weights_path": None,
            "status": "unavailable",
            "unavailable_reason": (
                "Shuttle weights not found — place yolov8m_shuttlecock.pt in "
                "backend/weights/ or set SHUTTLE_WEIGHTS"
            ),
        }
    return {
        "weights_found": True,
        "weights_path": weights,
        "status": "ok",
        "unavailable_reason": None,
    }


def resolve_shuttle_weights(explicit: str | None = None) -> str | None:
    if explicit and Path(explicit).is_file():
        return explicit
    env = os.environ.get("SHUTTLE_WEIGHTS", "").strip()
    if env and Path(env).is_file():
        return env
    candidate = _REPO_ROOT / "backend" / "weights" / "yolov8m_shuttlecock.pt"
    if candidate.is_file():
        return str(candidate)
    return None


class ShuttleKalman:
    """Pixel-space constant-velocity Kalman filter for shuttle overlay smoothing.

    Call ``update()`` when a detection is available, ``predict()`` when the
    shuttle is not detected.  ``predict()`` returns None once the gap exceeds
    ``gap_max`` frames, which triggers an internal reset so the next detection
    seeds a fresh trajectory.
    """

    def __init__(self, gap_max: int = 15) -> None:
        self._gap_max = gap_max
        self._x: np.ndarray | None = None  # [cx, cy, vx, vy]
        self._P: np.ndarray | None = None
        self._F = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        self._H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self._R = np.diag([4.0, 4.0]).astype(np.float64)
        self._I = np.eye(4, dtype=np.float64)
        self._predict_streak = 0

    def _scaled_Q(self) -> np.ndarray:
        base = np.diag([2.0, 2.0, 8.0, 8.0]).astype(np.float64)
        if self._x is None:
            return base
        speed = float((self._x[2] ** 2 + self._x[3] ** 2) ** 0.5)
        scale = min(4.0, 1.0 + speed / 25.0)
        return base * scale

    def reset(self) -> None:
        self._x = None
        self._P = None
        self._predict_streak = 0

    def update(self, cx: float, cy: float) -> tuple[float, float]:
        """Incorporate a detection; returns the filtered (cx, cy)."""
        z = np.array([cx, cy], dtype=np.float64)
        if self._x is None:
            self._x = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
            self._P = 10.0 * self._I.copy()
            self._predict_streak = 0
            return cx, cy
        x_pred = self._F @ self._x
        Q = self._scaled_Q()
        P_pred = self._F @ self._P @ self._F.T + Q
        K = P_pred @ self._H.T @ np.linalg.inv(self._H @ P_pred @ self._H.T + self._R)
        self._x = x_pred + K @ (z - self._H @ x_pred)
        self._P = (self._I - K @ self._H) @ P_pred
        self._predict_streak = 0
        return float(self._x[0]), float(self._x[1])

    def predict(self) -> tuple[float, float] | None:
        """Advance the filter one step without a measurement; returns None after ``gap_max`` frames."""
        if self._x is None:
            return None
        self._predict_streak += 1
        if self._predict_streak > self._gap_max:
            self.reset()
            return None
        self._x = self._F @ self._x
        if self._P is not None:
            Q = self._scaled_Q()
            self._P = self._F @ self._P @ self._F.T + Q
        return float(self._x[0]), float(self._x[1])


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
        raw_imgsz = os.environ.get("YOLO_IMGSZ", "").strip() or os.environ.get(
            "SHUTTLE_YOLO_IMGSZ", ""
        ).strip()
        self.imgsz = int(raw_imgsz) if raw_imgsz.isdigit() and int(raw_imgsz) > 0 else None

        if resolved is None:
            logger.warning(
                "Shuttle weights not found — rally detection disabled. "
                "Place yolov8m_shuttlecock.pt in backend/weights/ or set SHUTTLE_WEIGHTS."
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

        try:
            predict_kwargs: dict = {
                "conf": self.conf,
                "iou": self.iou,
                "device": self._device,
                "verbose": False,
            }
            if self.imgsz is not None:
                predict_kwargs["imgsz"] = self.imgsz
            result = self._model.predict(frame_bgr, **predict_kwargs)[0]
        except Exception as exc:
            logger.warning("Shuttle detect failed: %s", exc)
            return None
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
