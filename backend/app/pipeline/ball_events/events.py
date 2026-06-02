"""Heuristic target-player event detection from possession timeline."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from backend.app.pipeline.ball_events.identification import (
    has_target_identification,
    lock_confidence_for_id_type,
    log_frame_evaluation,
)
from backend.app.pipeline.ball_events.possession import PossessionFrame, PossessionState
from backend.app.pipeline.ball_events.timeline import FrameTimeline
from backend.app.pipeline.pitch_homography import PITCH_LENGTH_M, PITCH_WIDTH_M
from backend.app.schemas import PlayerEventCounts


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


GOAL_Y_MIN = (PITCH_WIDTH_M / 2.0) - (7.32 / 2.0)
GOAL_Y_MAX = (PITCH_WIDTH_M / 2.0) + (7.32 / 2.0)
GOAL_X_DEPTH = 5.5


@dataclass
class DetectedEvent:
    frame: int
    kind: str  # Pass | Shot | Goal | Drive
    confidence: str
    lock_confidence: str = "strong"  # strong | weak
    detection_phase: str | None = None  # locked | ocr | reid | track


def _ball_speed_m_s(ball, fps: float) -> float:
    if ball.vel_m_per_frame is None:
        return 0.0
    vx, vy = ball.vel_m_per_frame
    return math.hypot(vx, vy) * fps


def _infer_attack_right(target_xs: list[float]) -> bool:
    if not target_xs:
        return True
    return sum(target_xs) / len(target_xs) < PITCH_LENGTH_M / 2.0


def infer_attack_direction_label(
    timeline: list[FrameTimeline],
    locked_at_frame: int | None,
) -> str:
    target_xs = []
    for e in timeline:
        ok, _, _ = has_target_identification(e.target, locked_at_frame)
        if ok and e.target.target_m is not None:
            target_xs.append(e.target.target_m[0])
    return "left_to_right" if _infer_attack_right(target_xs) else "right_to_left"


def _toward_goal(ball_vx: float, ball_vy: float, attack_right: bool) -> bool:
    if attack_right:
        return ball_vx > 0.5
    return ball_vx < -0.5


def _in_goal_mouth(x: float, y: float, attack_right: bool) -> bool:
    if not (GOAL_Y_MIN <= y <= GOAL_Y_MAX):
        return False
    if attack_right:
        return x >= PITCH_LENGTH_M - GOAL_X_DEPTH
    return x <= GOAL_X_DEPTH


def detect_events(
    timeline: list[FrameTimeline],
    possession: list[PossessionFrame],
    *,
    fps: float,
    locked_at_frame: int | None,
    emitted_by_frame: dict[int, str] | None = None,
) -> list[DetectedEvent]:
    if len(timeline) != len(possession):
        raise ValueError("timeline and possession length mismatch")

    pass_min = _env_float("PASS_MIN_SPEED_M_S", 4.0)
    shot_min = _env_float("SHOT_MIN_SPEED_M_S", 8.0)
    debounce = _env_int("EVENT_DEBOUNCE_FRAMES", 20)
    goal_window = _env_int("GOAL_WINDOW_FRAMES", int(fps * 3))
    drive_min_frames = _env_int("DRIVE_MIN_FRAMES", 15)

    attack_right = _infer_attack_right(
        [
            e.target.target_m[0]
            for e in timeline
            if has_target_identification(e.target, locked_at_frame)[0]
            and e.target.target_m is not None
        ]
    )

    events: list[DetectedEvent] = []
    last_event_frame: dict[str, int] = {}
    possessed_states = (
        PossessionState.TARGET_POSSESSED,
        PossessionState.TARGET_POSSESSED_GAP,
    )

    prev_possessed = False
    drive_start: int | None = None

    for i, entry in enumerate(timeline):
        identified, lock_tier, id_type = has_target_identification(
            entry.target, locked_at_frame
        )
        if not identified:
            prev_possessed = False
            continue

        pf = possession[i]
        possessed = pf.state in possessed_states and pf.confidence != "low"

        ball_speed = _ball_speed_m_s(entry.ball, fps)
        dist_m_val = None
        if entry.target.target_m is not None and entry.ball.smooth_m is not None:
            dist_m_val = math.hypot(
                entry.target.target_m[0] - entry.ball.smooth_m[0],
                entry.target.target_m[1] - entry.ball.smooth_m[1],
            )

        if possessed and not prev_possessed:
            drive_start = entry.frame
        elif not possessed and prev_possessed and drive_start is not None:
            if entry.frame - drive_start >= drive_min_frames:
                kind = "Drive"
                if entry.frame - last_event_frame.get(kind, -9999) >= debounce:
                    events.append(
                        DetectedEvent(
                            frame=drive_start,
                            kind=kind,
                            confidence=pf.confidence,
                            lock_confidence=lock_tier or "weak",
                            detection_phase=id_type,
                        )
                    )
                    last_event_frame[kind] = entry.frame
                    if emitted_by_frame is not None:
                        emitted_by_frame[drive_start] = kind
            drive_start = None

        if prev_possessed and not possessed:
            speed = _ball_speed_m_s(entry.ball, fps)
            conf = pf.confidence
            if speed >= shot_min and entry.ball.vel_m_per_frame is not None:
                vx, vy = entry.ball.vel_m_per_frame
                vx_s = vx * fps
                if _toward_goal(vx_s, vy * fps, attack_right):
                    kind = "Shot"
                else:
                    kind = "Pass"
            elif speed >= pass_min:
                kind = "Pass"
            else:
                kind = None

            if kind is not None and entry.frame - last_event_frame.get(kind, -9999) >= debounce:
                ev_lock = lock_tier or lock_confidence_for_id_type(id_type) or "weak"
                events.append(
                    DetectedEvent(
                        frame=entry.frame,
                        kind=kind,
                        confidence=conf,
                        lock_confidence=ev_lock,
                        detection_phase=id_type,
                    )
                )
                last_event_frame[kind] = entry.frame
                if emitted_by_frame is not None:
                    emitted_by_frame[entry.frame] = kind
                log_frame_evaluation(
                    frame=entry.frame,
                    sample=entry.target,
                    locked_at_frame=locked_at_frame,
                    possessed=prev_possessed,
                    ball_speed_m_s=ball_speed,
                    dist_m=dist_m_val,
                    event_kind=kind,
                )

                if kind == "Shot":
                    for j in range(i, min(i + goal_window, len(timeline))):
                        later = timeline[j]
                        if later.ball.smooth_m is None:
                            continue
                        bx, by = later.ball.smooth_m
                        if _in_goal_mouth(bx, by, attack_right):
                            gkind = "Goal"
                            if later.frame - last_event_frame.get(gkind, -9999) >= debounce:
                                events.append(
                                    DetectedEvent(
                                        frame=later.frame,
                                        kind=gkind,
                                        confidence=conf,
                                        lock_confidence=ev_lock,
                                        detection_phase=id_type,
                                    )
                                )
                                last_event_frame[gkind] = later.frame
                                if emitted_by_frame is not None:
                                    emitted_by_frame[later.frame] = gkind
                            break

        prev_possessed = possessed

    return events


def aggregate_counts(events: list[DetectedEvent]) -> PlayerEventCounts:
    counts = {"Pass": 0, "Shot": 0, "Goal": 0, "Drive": 0}
    for ev in events:
        if ev.kind in counts:
            counts[ev.kind] += 1
    return PlayerEventCounts(**counts)
