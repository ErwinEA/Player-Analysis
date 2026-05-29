"""Target-centric ball possession finite-state machine."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from enum import Enum, auto

from backend.app.pipeline.ball_events.timeline import (
    FrameTimeline,
    dist_m,
    dist_px,
)


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


class PossessionState(Enum):
    NO_BALL = auto()
    TARGET_POSSESSED = auto()
    TARGET_POSSESSED_GAP = auto()
    UNKNOWN = auto()


@dataclass
class PossessionFrame:
    frame: int
    state: PossessionState
    confidence: str  # high | medium | low


class PossessionTracker:
    def __init__(self, fps: float) -> None:
        self.fps = max(fps, 1e-6)
        self._state = PossessionState.NO_BALL
        self._ball_missing = 0
        self._possess_radius = _env_float("POSSESS_RADIUS_M", 1.5)
        self._gap_radius = _env_float("GAP_POSSESS_RADIUS_M", 2.0)
        self._possess_radius_px = _env_float("POSSESS_RADIUS_PX", 90.0)
        self._gap_radius_px = _env_float("GAP_POSSESS_RADIUS_PX", 130.0)
        self._dribble_speed = _env_float("S_DRIBBLE_M_S", 8.0)
        self._ball_missing_max = _env_int("T_BALL_MISSING_FRAMES", 15)

    def reset(self) -> None:
        self._state = PossessionState.NO_BALL
        self._ball_missing = 0

    def _ball_speed_m_s(self, ball) -> float | None:
        if ball.vel_m_per_frame is None:
            return None
        vx, vy = ball.vel_m_per_frame
        return math.hypot(vx, vy) * self.fps

    def _proximity(self, entry: FrameTimeline) -> tuple[float, str] | None:
        """Distance for possession: image pixels when possible, else pitch metres."""
        ball = entry.ball
        target = entry.target
        if (
            target.target_px is not None
            and ball.cx_px > 0
            and ball.cy_px > 0
        ):
            return dist_px(target.target_px, ball.cx_px, ball.cy_px), "px"
        if ball.smooth_m is not None and target.target_m is not None:
            return dist_m(ball.smooth_m, target.target_m), "m"
        return None

    def _close(self, distance: float, units: str, *, gap: bool) -> bool:
        if distance < 0:
            return False
        if units == "px":
            return distance < (self._gap_radius_px if gap else self._possess_radius_px)
        return distance < (self._gap_radius if gap else self._possess_radius)

    def update(self, entry: FrameTimeline) -> PossessionFrame:
        ball = entry.ball
        target = entry.target

        if ball.smooth_m is None and (ball.predicted or ball.conf <= 0):
            self._ball_missing += 1
            if self._ball_missing > self._ball_missing_max:
                self._state = PossessionState.UNKNOWN
            return PossessionFrame(
                frame=entry.frame, state=self._state, confidence="low"
            )

        self._ball_missing = 0
        speed = self._ball_speed_m_s(ball) or 0.0
        target_m = target.target_m
        prox = self._proximity(entry)

        if target_m is None and prox is None:
            if self._state in (
                PossessionState.TARGET_POSSESSED,
                PossessionState.TARGET_POSSESSED_GAP,
            ):
                self._state = PossessionState.UNKNOWN
            else:
                self._state = PossessionState.NO_BALL
            return PossessionFrame(
                frame=entry.frame, state=self._state, confidence="low"
            )

        if prox is None:
            self._state = PossessionState.NO_BALL
            return PossessionFrame(
                frame=entry.frame, state=self._state, confidence="low"
            )

        proximity, units = prox

        if target.target_visible:
            if self._close(proximity, units, gap=False) and speed < self._dribble_speed:
                self._state = PossessionState.TARGET_POSSESSED
                return PossessionFrame(
                    frame=entry.frame, state=self._state, confidence="high"
                )
            if self._state == PossessionState.TARGET_POSSESSED and speed >= _env_float(
                "PASS_MIN_SPEED_M_S", 4.0
            ):
                self._state = PossessionState.UNKNOWN
                return PossessionFrame(
                    frame=entry.frame, state=self._state, confidence="high"
                )
            self._state = PossessionState.NO_BALL
            return PossessionFrame(
                frame=entry.frame, state=self._state, confidence="high"
            )

        # Target not visible — sticky gap possession
        if self._state in (
            PossessionState.TARGET_POSSESSED,
            PossessionState.TARGET_POSSESSED_GAP,
        ):
            if self._close(proximity, units, gap=True) and speed < self._dribble_speed:
                self._state = PossessionState.TARGET_POSSESSED_GAP
                return PossessionFrame(
                    frame=entry.frame, state=self._state, confidence="medium"
                )
            if speed >= _env_float("PASS_MIN_SPEED_M_S", 4.0):
                self._state = PossessionState.UNKNOWN
                return PossessionFrame(
                    frame=entry.frame, state=self._state, confidence="medium"
                )

        self._state = PossessionState.UNKNOWN
        return PossessionFrame(
            frame=entry.frame, state=self._state, confidence="low"
        )

    @property
    def possess_radius_m(self) -> float:
        return self._possess_radius

    @property
    def gap_possess_radius_m(self) -> float:
        return self._gap_radius

    @property
    def gap_max_frames(self) -> int:
        return self._ball_missing_max
