"""Plausibility filters for shuttle detections before Kalman / rally tracking."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.pipeline.badminton.shuttle_detector import ShuttleDetection
    from backend.app.pipeline.pitch_homography import PitchCalibration


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class ShuttleDetectionFilter:
    """Stateful gate: court bounds + max displacement vs last accepted detection."""

    calibration: PitchCalibration | None = None
    max_speed_px: float = field(
        default_factory=lambda: _env_float("SHUTTLE_MAX_SPEED_PX", 120.0)
    )
    max_speed_m_s: float = field(
        default_factory=lambda: _env_float("SHUTTLE_MAX_SPEED_M_S", 80.0)
    )
    court_margin_m: float = field(
        default_factory=lambda: _env_float("SHUTTLE_COURT_MARGIN_M", 0.5)
    )
    _last_px: tuple[float, float] | None = None
    _last_frame: int | None = None

    def reset(self) -> None:
        self._last_px = None
        self._last_frame = None

    def _on_court(self, cx: float, cy: float) -> bool:
        if self.calibration is None:
            return True
        try:
            x_m, y_m = self.calibration.pixel_to_meters(cx, cy)
        except ValueError:
            return False
        margin = self.court_margin_m
        length_m = self.calibration.pitch_length_m
        width_m = self.calibration.pitch_width_m
        return (
            -margin <= x_m <= length_m + margin
            and -margin <= y_m <= width_m + margin
        )

    def accept(self, det: ShuttleDetection, frame_idx: int, *, fps: float) -> bool:
        cx, cy = det.cx_px, det.cy_px
        if not self._on_court(cx, cy):
            return False

        if self._last_px is not None and self._last_frame is not None:
            dx = cx - self._last_px[0]
            dy = cy - self._last_px[1]
            dt_frames = max(1, frame_idx - self._last_frame)
            dist_px = (dx * dx + dy * dy) ** 0.5
            if dist_px / dt_frames > self.max_speed_px:
                return False
            if self.calibration is not None and fps > 0:
                dt_s = dt_frames / fps
                try:
                    x1, y1 = self.calibration.pixel_to_meters(
                        self._last_px[0], self._last_px[1]
                    )
                    x2, y2 = self.calibration.pixel_to_meters(cx, cy)
                except ValueError:
                    return False
                dist_m = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if dt_s > 0 and dist_m / dt_s > self.max_speed_m_s:
                    return False

        self._last_px = (cx, cy)
        self._last_frame = frame_idx
        return True
