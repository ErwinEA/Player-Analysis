"""Rally state machine driven by per-frame shuttle detections."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

RallyState = Literal["IDLE", "LIVE", "END"]
CourtSide = Literal["near", "far"]
WinnerSide = Literal["near", "far", "unknown"]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class RallyEvent:
    start_frame: int
    end_frame: int
    winner_side: WinnerSide
    locked_player_touched_last: bool = False


@dataclass
class RallyTracker:
    """IDLE -> LIVE -> END state machine over shuttle visibility.

    - IDLE -> LIVE after `live_frames` consecutive shuttle detections.
    - LIVE -> END after `gap_frames` frames without a shuttle.
    - END -> IDLE on the next update; the finished rally is appended to `events`.

    Win attribution (MVP heuristic): the rally is lost by the side where the
    shuttle was last seen (it landed there), so the opposite side wins.
    Assumes a standard broadcast view (near court = larger image Y below net).
    Inverted or rotated feeds need a future calibration flag to flip sides.
    """

    net_y_px: float
    fps: float = 30.0
    live_frames: int = field(default_factory=lambda: _env_int("SHUTTLE_LIVE_FRAMES", 2))
    gap_frames: int = field(default_factory=lambda: _env_int("SHUTTLE_GAP_FRAMES", 15))
    touch_radius_px: float = field(
        default_factory=lambda: _env_float("SHUTTLE_TOUCH_RADIUS_PX", 120.0)
    )
    touch_recent_frames: int = field(
        default_factory=lambda: _env_int("SHUTTLE_TOUCH_RECENT_FRAMES", 30)
    )

    state: RallyState = "IDLE"
    events: list[RallyEvent] = field(default_factory=list)

    _seen_streak: int = 0
    _gap_streak: int = 0
    _rally_start_frame: int | None = None
    _last_shuttle_frame: int | None = None
    _last_shuttle_y: float | None = None
    _last_locked_touch_frame: int | None = None

    def update(
        self,
        frame_idx: int,
        shuttle_px: tuple[float, float] | None,
        locked_foot_px: tuple[float, float] | None = None,
    ) -> RallyState:
        if self.state == "END":
            self.state = "IDLE"

        if shuttle_px is not None:
            self._seen_streak += 1
            self._gap_streak = 0
            self._last_shuttle_frame = frame_idx
            self._last_shuttle_y = shuttle_px[1]
            if locked_foot_px is not None:
                dx = shuttle_px[0] - locked_foot_px[0]
                dy = shuttle_px[1] - locked_foot_px[1]
                if (dx * dx + dy * dy) ** 0.5 <= self.touch_radius_px:
                    self._last_locked_touch_frame = frame_idx
        else:
            self._seen_streak = 0
            self._gap_streak += 1

        if self.state == "IDLE":
            if self._seen_streak >= self.live_frames:
                self.state = "LIVE"
                self._rally_start_frame = frame_idx - self.live_frames + 1
        elif self.state == "LIVE":
            if self._gap_streak >= self.gap_frames:
                self.state = "END"
                self._emit_rally(end_frame=self._last_shuttle_frame or frame_idx)

        return self.state

    def finalize(self, frame_idx: int) -> None:
        """Close an in-progress rally at end of video."""
        if self.state == "LIVE":
            self.state = "END"
            self._emit_rally(end_frame=self._last_shuttle_frame or frame_idx)

    def _emit_rally(self, *, end_frame: int) -> None:
        start = self._rally_start_frame if self._rally_start_frame is not None else end_frame
        winner: WinnerSide = "unknown"
        if self._last_shuttle_y is not None:
            # Shuttle landed on the side where it was last seen; that side loses.
            landed_near = self._last_shuttle_y > self.net_y_px
            winner = "far" if landed_near else "near"
        touched_last = (
            self._last_locked_touch_frame is not None
            and end_frame - self._last_locked_touch_frame <= self.touch_recent_frames
        )
        self.events.append(
            RallyEvent(
                start_frame=start,
                end_frame=end_frame,
                winner_side=winner,
                locked_player_touched_last=touched_last,
            )
        )
        self._rally_start_frame = None
        self._last_locked_touch_frame = None
