"""Per-frame join of ball state and target foot position."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from backend.app.pipeline.ball_events.ball_detector import BallState


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


@dataclass
class TargetFrameSample:
    frame: int
    target_m: tuple[float, float] | None
    target_visible: bool
    segment_source: str | None = None
    target_px: tuple[float, float] | None = None


@dataclass
class FrameTimeline:
    frame: int
    ball: BallState
    target: TargetFrameSample


class TargetFootTracker:
    """Track target foot in pitch metres with gap prediction (not gap_fill bbox)."""

    def __init__(self) -> None:
        self._last_m: tuple[float, float] | None = None
        self._last_px: tuple[float, float] | None = None
        self._last_v: tuple[float, float] = (0.0, 0.0)
        self._gap_frames = 0

    def reset(self) -> None:
        self._last_m = None
        self._last_px = None
        self._last_v = (0.0, 0.0)
        self._gap_frames = 0

    @property
    def last_px(self) -> tuple[float, float] | None:
        return self._last_px

    def observe(
        self,
        frame: int,
        foot_m: tuple[float, float] | None,
        *,
        visible: bool,
        segment_source: str | None,
        foot_px: tuple[float, float] | None = None,
    ) -> TargetFrameSample:
        gap_max = _env_int("TARGET_GAP_MAX_FRAMES", 22)

        if (
            visible
            and foot_m is not None
            and segment_source != "gap_fill"
        ):
            if self._last_m is not None:
                self._last_v = (
                    foot_m[0] - self._last_m[0],
                    foot_m[1] - self._last_m[1],
                )
            self._last_m = foot_m
            if foot_px is not None:
                self._last_px = foot_px
            self._gap_frames = 0
            return TargetFrameSample(
                frame=frame,
                target_m=foot_m,
                target_visible=True,
                segment_source=segment_source,
                target_px=foot_px,
            )

        if (
            foot_m is not None
            and foot_px is not None
            and segment_source == "bbox_track"
        ):
            self._last_m = foot_m
            self._last_px = foot_px
            self._gap_frames = 0
            return TargetFrameSample(
                frame=frame,
                target_m=foot_m,
                target_visible=True,
                segment_source=segment_source,
                target_px=foot_px,
            )

        if self._last_m is None:
            return TargetFrameSample(
                frame=frame,
                target_m=None,
                target_visible=False,
                segment_source=segment_source,
            )

        self._gap_frames += 1
        if self._gap_frames > gap_max:
            return TargetFrameSample(
                frame=frame,
                target_m=None,
                target_visible=False,
                segment_source=segment_source,
            )

        pred = (
            self._last_m[0] + self._last_v[0],
            self._last_m[1] + self._last_v[1],
        )
        return TargetFrameSample(
            frame=frame,
            target_m=pred,
            target_visible=False,
            segment_source=segment_source,
            target_px=foot_px if foot_px is not None else self._last_px,
        )


def dist_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dist_px(
    target_px: tuple[float, float],
    ball_cx_px: float,
    ball_cy_px: float,
) -> float:
    return math.hypot(target_px[0] - ball_cx_px, target_px[1] - ball_cy_px)


def build_timeline(
    ball_states: list[BallState],
    target_samples: list[TargetFrameSample],
) -> list[FrameTimeline]:
    target_by_frame = {t.frame: t for t in target_samples}
    out: list[FrameTimeline] = []
    for ball in ball_states:
        target = target_by_frame.get(
            ball.frame,
            TargetFrameSample(
                frame=ball.frame,
                target_m=None,
                target_visible=False,
            ),
        )
        out.append(FrameTimeline(frame=ball.frame, ball=ball, target=target))
    return out
