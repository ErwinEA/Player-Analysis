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
from backend.app.pipeline.ball_events.timeline import FrameTimeline, dist_m
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
    prev_possessed_track = False
    pending_release: tuple[int, FrameTimeline, PossessionFrame] | None = None
    drive_start: int | None = None

    def _emit_release(
        rel_i: int,
        rel_entry: FrameTimeline,
        rel_pf: PossessionFrame,
        lock_tier: str | None,
        id_type: str | None,
    ) -> None:
        speed = _ball_speed_m_s(rel_entry.ball, fps)
        conf = rel_pf.confidence
        if speed >= shot_min and rel_entry.ball.vel_m_per_frame is not None:
            vx, vy = rel_entry.ball.vel_m_per_frame
            vx_s = vx * fps
            if _toward_goal(vx_s, vy * fps, attack_right):
                kind = "Shot"
            else:
                kind = "Pass"
        elif speed >= pass_min:
            kind = "Pass"
        else:
            kind = None

        if kind is None or rel_entry.frame - last_event_frame.get(kind, -9999) < debounce:
            return

        ev_lock = lock_tier or lock_confidence_for_id_type(id_type) or "weak"
        events.append(
            DetectedEvent(
                frame=rel_entry.frame,
                kind=kind,
                confidence=conf,
                lock_confidence=ev_lock,
                detection_phase=id_type,
            )
        )
        last_event_frame[kind] = rel_entry.frame
        if emitted_by_frame is not None:
            emitted_by_frame[rel_entry.frame] = kind
        rel_dist_m = None
        if rel_entry.target.target_m is not None and rel_entry.ball.smooth_m is not None:
            rel_dist_m = math.hypot(
                rel_entry.target.target_m[0] - rel_entry.ball.smooth_m[0],
                rel_entry.target.target_m[1] - rel_entry.ball.smooth_m[1],
            )
        log_frame_evaluation(
            frame=rel_entry.frame,
            sample=rel_entry.target,
            locked_at_frame=locked_at_frame,
            possessed=True,
            ball_speed_m_s=speed,
            dist_m=rel_dist_m,
            event_kind=kind,
        )

        if kind != "Shot":
            return
        for j in range(rel_i, min(rel_i + goal_window, len(timeline))):
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

    def _emit_drive(
        end_frame: int,
        pf: PossessionFrame,
        lock_tier: str | None,
        id_type: str | None,
    ) -> None:
        nonlocal drive_start
        if drive_start is None:
            return
        if end_frame - drive_start < drive_min_frames:
            drive_start = None
            return
        kind = "Drive"
        if end_frame - last_event_frame.get(kind, -9999) >= debounce:
            events.append(
                DetectedEvent(
                    frame=drive_start,
                    kind=kind,
                    confidence=pf.confidence,
                    lock_confidence=lock_tier or lock_confidence_for_id_type(id_type) or "weak",
                    detection_phase=id_type,
                )
            )
            last_event_frame[kind] = end_frame
            if emitted_by_frame is not None:
                emitted_by_frame[drive_start] = kind
        drive_start = None

    for i, entry in enumerate(timeline):
        pf = possession[i]
        possessed = pf.state in possessed_states and pf.confidence != "low"

        if prev_possessed_track and not possessed:
            pending_release = (i, entry, pf)
        prev_possessed_track = possessed

        identified, lock_tier, id_type = has_target_identification(
            entry.target, locked_at_frame
        )
        if not identified:
            continue

        if pending_release is not None:
            rel_i, rel_entry, rel_pf = pending_release
            _emit_release(rel_i, rel_entry, rel_pf, lock_tier, id_type)
            pending_release = None

        if possessed and not prev_possessed:
            drive_start = entry.frame
        elif not possessed and prev_possessed:
            _emit_drive(entry.frame, pf, lock_tier, id_type)

        prev_possessed = possessed

    if pending_release is not None:
        rel_i, rel_entry, rel_pf = pending_release
        flush_tier: str | None = None
        flush_id: str | None = None
        for later in timeline[rel_i:]:
            ok, tier, phase = has_target_identification(
                later.target, locked_at_frame
            )
            if ok:
                flush_tier, flush_id = tier, phase
                break
        _emit_release(rel_i, rel_entry, rel_pf, flush_tier, flush_id)

    if prev_possessed and drive_start is not None and timeline:
        last_entry = timeline[-1]
        _, end_lock, end_id = has_target_identification(
            last_entry.target, locked_at_frame
        )
        _emit_drive(last_entry.frame + 1, possession[-1], end_lock, end_id)

    return events


def compute_drive_contact_m(
    timeline: list[FrameTimeline],
    possession: list[PossessionFrame],
    *,
    locked_at_frame: int | None,
) -> float:
    """Metres travelled by the target while in qualifying drive possession segments."""
    if len(timeline) != len(possession):
        raise ValueError("timeline and possession length mismatch")

    drive_min_frames = _env_int("DRIVE_MIN_FRAMES", 15)
    possessed_states = (
        PossessionState.TARGET_POSSESSED,
        PossessionState.TARGET_POSSESSED_GAP,
    )

    total_m = 0.0
    segment_m = 0.0
    segment_start_frame: int | None = None
    prev_possessed = False
    prev_target_m: tuple[float, float] | None = None

    def _flush_segment(end_frame: int) -> None:
        nonlocal total_m, segment_m, segment_start_frame, prev_target_m
        if segment_start_frame is not None and end_frame - segment_start_frame >= drive_min_frames:
            total_m += segment_m
        segment_m = 0.0
        segment_start_frame = None
        prev_target_m = None

    for entry, pf in zip(timeline, possession, strict=True):
        identified, _, _ = has_target_identification(entry.target, locked_at_frame)
        if not identified:
            continue

        possessed = (
            pf.state in possessed_states and pf.confidence != "low"
        )
        target_m = entry.target.target_m

        if possessed and not prev_possessed:
            segment_start_frame = entry.frame
            prev_target_m = target_m
        elif possessed and prev_possessed:
            if target_m is not None and prev_target_m is not None:
                segment_m += dist_m(prev_target_m, target_m)
            prev_target_m = target_m
        elif prev_possessed:
            _flush_segment(entry.frame)

        prev_possessed = possessed

    if prev_possessed and timeline:
        _flush_segment(timeline[-1].frame + 1)

    return total_m


def aggregate_counts(events: list[DetectedEvent]) -> PlayerEventCounts:
    counts = {"Pass": 0, "Shot": 0, "Goal": 0, "Drive": 0}
    for ev in events:
        if ev.kind in counts:
            counts[ev.kind] += 1
    return PlayerEventCounts(**counts)
