"""YOLO ball detection with pitch-space Kalman smoothing."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from backend.app.pipeline.pitch_homography import PitchCalibration

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def resolve_ball_weights(explicit: str | None = None) -> str | None:
    if explicit and Path(explicit).is_file():
        return explicit
    env = os.environ.get("BALL_WEIGHTS", "").strip()
    if env and Path(env).is_file():
        return env
    for candidate in (
        _REPO_ROOT / "detetction_test" / "weights" / "yolov8n_ball.pt",
        _REPO_ROOT / "backend" / "weights" / "yolov8n_ball.pt",
    ):
        if candidate.is_file():
            return str(candidate)
    return None


@dataclass(frozen=True)
class BallDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def cx_px(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy_px(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def center_px(self) -> tuple[float, float]:
        return (self.cx_px, self.cy_px)


@dataclass(frozen=True)
class BallState:
    frame: int
    cx_px: float
    cy_px: float
    conf: float
    pitch_m: tuple[float, float] | None
    smooth_m: tuple[float, float] | None
    vel_m_per_frame: tuple[float, float] | None
    predicted: bool

    @property
    def speed_m_s(self) -> float | None:
        if self.vel_m_per_frame is None:
            return None
        vx, vy = self.vel_m_per_frame
        return float(np.hypot(vx, vy))

    def speed_m_s_at_fps(self, fps: float) -> float | None:
        base = self.speed_m_s
        if base is None or fps <= 0:
            return None
        return base * fps


def _empty_state(frame: int) -> BallState:
    return BallState(
        frame=frame,
        cx_px=0.0,
        cy_px=0.0,
        conf=0.0,
        pitch_m=None,
        smooth_m=None,
        vel_m_per_frame=None,
        predicted=False,
    )


class _PitchKalman:
    """Constant-velocity Kalman filter in pitch metres."""

    def __init__(self, q: float = 0.5, r: float = 1.0) -> None:
        self._q = q
        self._r = r
        self._x: np.ndarray | None = None  # [x, y, vx, vy]
        self._P: np.ndarray | None = None
        self._F = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        self._H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self._Q = np.diag([q, q, 2 * q, 2 * q]).astype(np.float64)
        self._R = np.diag([r, r]).astype(np.float64)
        self._I = np.eye(4, dtype=np.float64)

    def reset(self) -> None:
        self._x = None
        self._P = None

    def _init(self, x_m: float, y_m: float) -> None:
        self._x = np.array([x_m, y_m, 0.0, 0.0], dtype=np.float64)
        self._P = 10.0 * self._I.copy()

    def update(self, x_m: float, y_m: float) -> tuple[float, float, float, float]:
        z = np.array([x_m, y_m], dtype=np.float64)
        if self._x is None:
            self._init(x_m, y_m)
            return float(self._x[0]), float(self._x[1]), 0.0, 0.0

        assert self._P is not None
        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q
        S = self._H @ P_pred @ self._H.T + self._R
        K = P_pred @ self._H.T @ np.linalg.inv(S)
        y = z - self._H @ x_pred
        self._x = x_pred + K @ y
        self._P = (self._I - K @ self._H) @ P_pred
        return (
            float(self._x[0]),
            float(self._x[1]),
            float(self._x[2]),
            float(self._x[3]),
        )

    def predict(self) -> tuple[float, float, float, float]:
        if self._x is None:
            return 0.0, 0.0, 0.0, 0.0
        assert self._P is not None
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return (
            float(self._x[0]),
            float(self._x[1]),
            float(self._x[2]),
            float(self._x[3]),
        )


class BallDetector:
    """YOLOv8 ball detector with optional pitch Kalman smoothing."""

    def __init__(
        self,
        weights: str | None = None,
        conf: float | None = None,
        iou: float | None = None,
        fps: float = 25.0,
    ) -> None:
        self.conf = conf if conf is not None else _env_float("BALL_CONF", 0.25)
        self.iou = iou if iou is not None else _env_float("BALL_IOU", 0.5)
        self._gap_max = _env_int("BALL_KALMAN_GAP_MAX_FRAMES", 22)
        self._kalman_q = _env_float("BALL_KALMAN_Q", 0.5)
        self._kalman_r = _env_float("BALL_KALMAN_R", 1.0)
        # Physical ceiling on ball motion: a projected detection implying a larger
        # frame-to-frame jump is a far-field false positive / homography blow-up and
        # is rejected so it can't poison the Kalman velocity (see BALL_MAX_SPEED_M_S).
        max_speed = _env_float("BALL_MAX_SPEED_M_S", 36.0)
        self._max_jump_m = max_speed / fps if fps > 0 else max_speed
        self._outlier_max = _env_int("BALL_OUTLIER_MAX_FRAMES", 5)

        resolved = resolve_ball_weights(weights)
        self.weights: str | None = resolved
        self.available = False
        self._model = None
        self._kalman = _PitchKalman(q=self._kalman_q, r=self._kalman_r)
        self._predict_streak = 0
        self._reject_streak = 0
        self._last_cx_px = 0.0
        self._last_cy_px = 0.0

        if resolved is None:
            logger.warning(
                "Ball weights not found — ball events disabled. "
                "Place yolov8n_ball.pt in detetction_test/weights/ or set BALL_WEIGHTS."
            )
            return

        try:
            from ultralytics import YOLO

            self._model = YOLO(resolved)
            self.available = True
            self.weights = resolved
        except Exception as exc:
            logger.warning("Failed to load ball YOLO weights %s: %s", resolved, exc)
            self.available = False

    def reset(self) -> None:
        self._kalman.reset()
        self._predict_streak = 0
        self._reject_streak = 0
        self._last_cx_px = 0.0
        self._last_cy_px = 0.0

    def detect(
        self,
        frame_bgr: NDArray[np.uint8],
        near_px: tuple[float, float] | None = None,
    ) -> BallDetection | None:
        if not self.available or self._model is None:
            return None

        result = self._model.predict(
            frame_bgr,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        centers = np.column_stack(
            ((boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0)
        )
        if near_px is not None:
            anchor_max = _env_float("BALL_ANCHOR_MAX_PX", 350.0)
            dists = np.hypot(
                centers[:, 0] - near_px[0], centers[:, 1] - near_px[1]
            )
            in_range = np.where(dists <= anchor_max)[0]
            if len(in_range) > 0:
                best_i = int(in_range[np.argmax(scores[in_range])])
            else:
                best_i = int(np.argmin(dists))
        else:
            best_i = int(np.argmax(scores))
        box = boxes[best_i]
        return BallDetection(
            x1=float(box[0]),
            y1=float(box[1]),
            x2=float(box[2]),
            y2=float(box[3]),
            conf=float(scores[best_i]),
        )

    def update_pitch(
        self,
        frame_idx: int,
        detection: BallDetection | None,
        calibration: PitchCalibration,
    ) -> BallState:
        if not self.available:
            return _empty_state(frame_idx)

        if detection is not None:
            try:
                pitch_m = calibration.pixel_to_meters(
                    detection.cx_px, detection.cy_px
                )
            except ValueError:
                pitch_m = None
        else:
            pitch_m = None

        # Reject detections that imply impossible motion (far-field false positives /
        # homography blow-ups). Coast on prediction so the velocity stays sane; only
        # re-acquire if a new location persists past BALL_OUTLIER_MAX_FRAMES.
        if pitch_m is not None and self._kalman._x is not None:
            jump = float(
                np.hypot(
                    pitch_m[0] - self._kalman._x[0],
                    pitch_m[1] - self._kalman._x[1],
                )
            )
            if jump > self._max_jump_m:
                self._reject_streak += 1
                if self._reject_streak <= self._outlier_max:
                    return self._coast(frame_idx)
                self._kalman.reset()
                self._reject_streak = 0

        if pitch_m is not None:
            self._predict_streak = 0
            self._reject_streak = 0
            if detection is not None:
                self._last_cx_px = detection.cx_px
                self._last_cy_px = detection.cy_px
            sx, sy, vx, vy = self._kalman.update(pitch_m[0], pitch_m[1])
            return BallState(
                frame=frame_idx,
                cx_px=detection.cx_px if detection else self._last_cx_px,
                cy_px=detection.cy_px if detection else self._last_cy_px,
                conf=detection.conf if detection else 0.0,
                pitch_m=pitch_m,
                smooth_m=(sx, sy),
                vel_m_per_frame=(vx, vy),
                predicted=False,
            )

        return self._coast(frame_idx)

    def _coast(self, frame_idx: int) -> BallState:
        """Advance the filter on prediction during a missing/rejected detection."""
        if self._kalman._x is None:
            return _empty_state(frame_idx)

        self._predict_streak += 1
        if self._predict_streak > self._gap_max:
            logger.debug(
                "Ball Kalman gap exceeded (%s frames); resetting filter at frame %s",
                self._gap_max,
                frame_idx,
            )
            self._kalman.reset()
            self._predict_streak = 0
            self._reject_streak = 0
            return _empty_state(frame_idx)

        sx, sy, vx, vy = self._kalman.predict()
        return BallState(
            frame=frame_idx,
            cx_px=self._last_cx_px,
            cy_px=self._last_cy_px,
            conf=0.0,
            pitch_m=None,
            smooth_m=(sx, sy),
            vel_m_per_frame=(vx, vy),
            predicted=True,
        )

    def update_pixels(
        self,
        frame_idx: int,
        detection: BallDetection | None,
    ) -> BallState:
        if detection is None:
            return _empty_state(frame_idx)
        return BallState(
            frame=frame_idx,
            cx_px=detection.cx_px,
            cy_px=detection.cy_px,
            conf=detection.conf,
            pitch_m=None,
            smooth_m=None,
            vel_m_per_frame=None,
            predicted=False,
        )
