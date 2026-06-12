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
    serving_side: WinnerSide = "near"
    locked_player_touched_last: bool = False


@dataclass
class RallyTracker:
    """IDLE -> LIVE -> END state machine over shuttle motion.

    - IDLE -> LIVE after `live_frames` consecutive detections that include
      serve/flight motion (shuttle not sitting still on the court).
    - LIVE -> END when the shuttle lands: vertical speed collapses after a
      period of in-flight motion, or after `gap_frames` without a raw detection
      once flight was observed.
    - END -> IDLE on the next update; the finished rally is appended to `events`.

  Win attribution (MVP heuristic): the rally is lost by the side where the
    shuttle was last seen (it landed there), so the opposite side wins.
    Assumes a standard broadcast view (near court = larger image Y below net).
    Inverted or rotated feeds need a future calibration flag to flip sides.

  Tunables (environment variables):
    SHUTTLE_LIVE_FRAMES — consecutive detections to start a rally (default 1)
    SHUTTLE_GAP_FRAMES — miss streak after flight to end rally (default 8)
    SHUTTLE_START_GAP_MAX — miss frames before serve-motion latch resets (default 2)
    SHUTTLE_FLIGHT_VY_PX — min |vertical px/frame| for in-flight (default 4)
    SHUTTLE_MIN_FLIGHT_FRAMES — flight frames before landing can end (default 2)
    SHUTTLE_LAND_VY_PX — max |vy| when shuttle has landed (default 2.5)
    SHUTTLE_LAND_STABLE_FRAMES — low-velocity frames to confirm landing (default 3)
    SHUTTLE_SERVE_SPEED_PX — min px/frame motion to count as serve/hit (default 1.5)
    SHUTTLE_POST_LAND_COOLDOWN_FRAMES — idle frames after rally end (default 6)
    SHUTTLE_TOUCH_RADIUS_PX, SHUTTLE_TOUCH_RECENT_FRAMES — locked-player touch
    """

    net_y_px: float
    fps: float = 30.0
    live_frames: int = field(default_factory=lambda: _env_int("SHUTTLE_LIVE_FRAMES", 1))
    gap_frames: int = field(default_factory=lambda: _env_int("SHUTTLE_GAP_FRAMES", 8))
    flight_vy_px: float = field(
        default_factory=lambda: _env_float("SHUTTLE_FLIGHT_VY_PX", 4.0)
    )
    min_flight_frames: int = field(
        default_factory=lambda: _env_int("SHUTTLE_MIN_FLIGHT_FRAMES", 2)
    )
    land_vy_px: float = field(
        default_factory=lambda: _env_float("SHUTTLE_LAND_VY_PX", 2.5)
    )
    land_stable_frames: int = field(
        default_factory=lambda: _env_int("SHUTTLE_LAND_STABLE_FRAMES", 3)
    )
    serve_speed_px: float = field(
        default_factory=lambda: _env_float("SHUTTLE_SERVE_SPEED_PX", 1.5)
    )
    start_gap_max: int = field(
        default_factory=lambda: _env_int("SHUTTLE_START_GAP_MAX", 2)
    )
    post_land_cooldown_frames: int = field(
        default_factory=lambda: _env_int("SHUTTLE_POST_LAND_COOLDOWN_FRAMES", 6)
    )
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
    _prev_shuttle_y: float | None = None
    _prev_shuttle_x: float | None = None
    _flight_frames: int = 0
    _land_stable_streak: int = 0
    _idle_motion_seen: bool = False
    _post_land_cooldown: int = 0
    _last_locked_touch_frame: int | None = None
    _serving_side: WinnerSide = "near"

    def update(
        self,
        frame_idx: int,
        shuttle_px: tuple[float, float] | None,
        locked_foot_px: tuple[float, float] | None = None,
    ) -> RallyState:
        if self.state == "END":
            self.state = "IDLE"

        if self._post_land_cooldown > 0:
            self._post_land_cooldown -= 1

        if shuttle_px is not None:
            self._seen_streak += 1
            self._gap_streak = 0
            self._last_shuttle_frame = frame_idx
            x, y = shuttle_px
            speed = self._shuttle_speed(x, y)
            vy = self._shuttle_vy(y)

            if locked_foot_px is not None:
                dx = x - locked_foot_px[0]
                dy = y - locked_foot_px[1]
                if (dx * dx + dy * dy) ** 0.5 <= self.touch_radius_px:
                    self._last_locked_touch_frame = frame_idx

            if self.state == "IDLE":
                if speed >= self.serve_speed_px:
                    self._idle_motion_seen = True
            elif self.state == "LIVE":
                self._update_flight_land_signals(speed, vy)

            self._last_shuttle_y = y
            self._prev_shuttle_x = x
            self._prev_shuttle_y = y
        else:
            self._seen_streak = 0
            self._gap_streak += 1
            if self._gap_streak > self.start_gap_max:
                self._idle_motion_seen = False
            self._prev_shuttle_x = None
            self._prev_shuttle_y = None

        if self.state == "IDLE":
            if (
                self._post_land_cooldown == 0
                and self._seen_streak >= self.live_frames
                and self._idle_motion_seen
            ):
                self.state = "LIVE"
                self._rally_start_frame = frame_idx - self.live_frames + 1
                # Serve/hit motion that opened the rally counts as initial flight.
                self._flight_frames = (
                    self.min_flight_frames if self._idle_motion_seen else 0
                )
                self._land_stable_streak = 0
        elif self.state == "LIVE":
            landed = (
                self._flight_frames >= self.min_flight_frames
                and self._land_stable_streak >= self.land_stable_frames
            )
            gap_after_flight = (
                self._flight_frames >= self.min_flight_frames
                and self._gap_streak >= self.gap_frames
            )
            if landed or gap_after_flight:
                self.state = "END"
                self._emit_rally(end_frame=self._last_shuttle_frame or frame_idx)
                self._post_land_cooldown = self.post_land_cooldown_frames
                self._flight_frames = 0
                self._land_stable_streak = 0
                self._idle_motion_seen = False

        return self.state

    def finalize(self, frame_idx: int) -> None:
        """Close an in-progress rally at end of video."""
        if self.state == "LIVE":
            self.state = "END"
            self._emit_rally(end_frame=self._last_shuttle_frame or frame_idx)
            self._post_land_cooldown = self.post_land_cooldown_frames

    def _shuttle_vy(self, y: float) -> float:
        if self._prev_shuttle_y is None:
            return 0.0
        return y - self._prev_shuttle_y

    def _shuttle_speed(self, x: float, y: float) -> float:
        if self._prev_shuttle_x is None or self._prev_shuttle_y is None:
            return 0.0
        dx = x - self._prev_shuttle_x
        dy = y - self._prev_shuttle_y
        return (dx * dx + dy * dy) ** 0.5

    def _update_flight_land_signals(self, speed: float, vy: float) -> None:
        if speed >= self.flight_vy_px:
            self._flight_frames += 1
            self._land_stable_streak = 0
        elif self._flight_frames >= self.min_flight_frames:
            if speed <= self.land_vy_px and abs(vy) <= self.land_vy_px:
                self._land_stable_streak += 1
            else:
                self._land_stable_streak = 0

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
        serving = self._serving_side
        self.events.append(
            RallyEvent(
                start_frame=start,
                end_frame=end_frame,
                winner_side=winner,
                serving_side=serving,
                locked_player_touched_last=touched_last,
            )
        )
        self._serving_side = "far" if serving == "near" else "near"
        self._rally_start_frame = None
        self._last_locked_touch_frame = None
