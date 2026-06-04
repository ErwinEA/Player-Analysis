"""Orchestrate timeline → possession → event counts."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass

from backend.app.pipeline.ball_events.ball_detector import BallState
from backend.app.pipeline.ball_events.events import (
    DetectedEvent,
    aggregate_counts,
    compute_drive_contact_m,
    detect_events,
    infer_attack_direction_label,
)
from backend.app.pipeline.ball_events.identification import has_target_identification
from backend.app.pipeline.ball_events.possession import PossessionTracker
from backend.app.pipeline.ball_events.timeline import (
    TargetFrameSample,
    build_timeline,
    dist_m,
)
from backend.app.schemas import PlayerEventCounts

logger = logging.getLogger(__name__)


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


def _debug_enabled() -> bool:
    return os.environ.get("BALL_EVENTS_DEBUG", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


@dataclass
class FrameDiag:
    frame: int
    t_s: float
    target_visible: bool
    target_m: list[float] | None
    ball_smooth_m: list[float] | None
    ball_predicted: bool
    dist_m: float | None
    ball_speed_m_s: float | None
    fsm_state: str
    event_emitted: str | None


def run_events(
    ball_states: list[BallState],
    target_samples: list[TargetFrameSample],
    *,
    fps: float,
    locked_at_frame: int | None,
) -> tuple[PlayerEventCounts, int, list[DetectedEvent], float]:
    """
    Returns (event_counts, ball_samples, detected_events, drive_contact_m) where
    ball_samples = frames with a raw detection.
    """
    empty = PlayerEventCounts()
    ball_samples = _count_detections(ball_states)
    has_soft_id = any(s.identification_type for s in target_samples)
    if locked_at_frame is None and not has_soft_id:
        return empty, ball_samples, [], 0.0

    timeline = build_timeline(ball_states, target_samples)
    if not timeline:
        return empty, 0, [], 0.0

    pass_min = _env_float("PASS_MIN_SPEED_M_S", 4.0)
    shot_min = _env_float("SHOT_MIN_SPEED_M_S", 8.0)
    possess_r = _env_float("POSSESS_RADIUS_M", 2.0)
    gap_r = _env_float("GAP_POSSESS_RADIUS_M", 2.5)
    gap_max = _env_int("TARGET_GAP_MAX_FRAMES", 22)
    attack_dir = infer_attack_direction_label(timeline, locked_at_frame)
    logger.info(
        "[BallEvents] Thresholds — possess_r=%.2fm  gap_r=%.2fm  gap_max=%sf",
        possess_r,
        gap_r,
        gap_max,
    )
    logger.info(
        "[BallEvents] Thresholds — pass_speed=%.2fm/s  shot_speed=%.2fm/s  attack_dir=%s",
        pass_min,
        shot_min,
        attack_dir,
    )
    logger.info(
        "[BallEvents] Input — frames=%s  ball_samples=%s  locked_at=%s",
        len(timeline),
        ball_samples,
        locked_at_frame,
    )

    debug_on = _debug_enabled()
    debug_frames: list[FrameDiag] | None = [] if debug_on else None

    tracker = PossessionTracker(fps=fps)
    possession = []
    for entry in timeline:
        pf = tracker.update(entry)
        possession.append(pf)
        if debug_frames is None:
            continue
        if not has_target_identification(entry.target, locked_at_frame)[0]:
            continue
        target_m = entry.target.target_m
        ball_m = entry.ball.smooth_m
        d = (
            dist_m(target_m, ball_m)
            if target_m is not None and ball_m is not None
            else None
        )
        speed = (
            math.hypot(*entry.ball.vel_m_per_frame) * fps
            if entry.ball.vel_m_per_frame is not None
            else None
        )
        debug_frames.append(
            FrameDiag(
                frame=entry.frame,
                t_s=entry.frame / fps if fps > 0 else 0.0,
                target_visible=entry.target.target_visible,
                target_m=list(target_m) if target_m is not None else None,
                ball_smooth_m=list(ball_m) if ball_m is not None else None,
                ball_predicted=entry.ball.predicted,
                dist_m=d,
                ball_speed_m_s=speed,
                fsm_state="".join(
                    p.capitalize() for p in pf.state.name.lower().split("_")
                ),
                event_emitted=None,
            )
        )
    emitted_by_frame: dict[int, str] | None = {} if debug_on else None
    events = detect_events(
        timeline,
        possession,
        fps=fps,
        locked_at_frame=locked_at_frame,
        emitted_by_frame=emitted_by_frame,
    )
    counts = aggregate_counts(events)
    drive_contact_m = compute_drive_contact_m(
        timeline,
        possession,
        locked_at_frame=locked_at_frame,
    )
    for ev in events:
        logger.info(
            "[BallEvents] Event — frame=%s kind=%s lock_confidence=%s phase=%s",
            ev.frame,
            ev.kind,
            ev.lock_confidence,
            ev.detection_phase or "-",
        )
    # #region agent log
    _log_event_summary(
        timeline=timeline,
        possession=possession,
        locked_at_frame=locked_at_frame,
        counts=counts,
        ball_samples=ball_samples,
    )
    # #endregion
    if debug_frames is not None:
        emitted_by_frame = emitted_by_frame or {}
        for d in debug_frames:
            d.event_emitted = emitted_by_frame.get(d.frame)
        _write_diagnostic(
            debug_frames=debug_frames,
            total_frames=len(timeline),
            fps=fps,
            locked_at_frame=locked_at_frame,
            possess_radius_m=possess_r,
            gap_possess_radius_m=gap_r,
            gap_max_frames=gap_max,
            pass_min_speed_m_s=pass_min,
            shot_min_speed_m_s=shot_min,
            ball_samples=ball_samples,
            events_found=counts.model_dump(),
        )
    return counts, ball_samples, events, drive_contact_m


@dataclass
class LockEpoch:
    """A contiguous lock window bounded by scene cuts.

    ball_states / target_samples hold the per-frame samples accumulated during the
    window; locked_at_frame is the frame the lock was acquired (None if the window
    never produced a lock and so contributes no events).
    """

    start_frame: int
    end_frame: int
    locked_at_frame: int | None
    track_id: int | None
    ball_states: list[BallState]
    target_samples: list[TargetFrameSample]


def _epoch_quality(epoch: LockEpoch) -> tuple[int, int]:
    """Score an epoch as (on-pitch visible identified frames, identified frame count)."""
    from backend.app.pipeline.pitch_homography import is_on_pitch

    visible = 0
    identified = 0
    for s in epoch.target_samples:
        ok, _, _ = has_target_identification(s, epoch.locked_at_frame)
        if not ok:
            continue
        identified += 1
        if (
            s.target_visible
            and s.target_m is not None
            and is_on_pitch(s.target_m[0], s.target_m[1])
        ):
            visible += 1
    return (visible, identified)


def _epoch_analyzable(epoch: LockEpoch, *, min_visible: int) -> bool:
    visible, identified = _epoch_quality(epoch)
    if identified == 0:
        return False
    return visible >= min_visible


def run_events_multi(
    epochs: list[LockEpoch],
    *,
    fps: float,
) -> tuple[PlayerEventCounts, int, LockEpoch | None, list[DetectedEvent], float]:
    """Run the possession FSM per lock epoch and aggregate the results.

    Event counts are summed across every epoch clearing a minimum-quality bar
    (identified target + EPOCH_MIN_VISIBLE_FRAMES on-pitch visible frames) so a
    player appearing across several broadcast shots is not undercounted. Epochs
    with only soft identification (pre-lock OCR/ReID/track) are included. The
    single best epoch (most on-pitch visible frames, tie-break longest) is returned
    for single-track artefacts (heatmap / movement) that cannot merge across re-locks.
    """
    min_visible = _env_int("EPOCH_MIN_VISIBLE_FRAMES", 5)
    total_samples = sum(_count_detections(ep.ball_states) for ep in epochs)
    aggregated = PlayerEventCounts()
    all_events: list[DetectedEvent] = []
    drive_contact_m = 0.0
    best: LockEpoch | None = None
    best_score = (-1, -1)
    counted = 0
    for ep in epochs:
        if not _epoch_analyzable(ep, min_visible=min_visible):
            continue
        counts, _, events, epoch_drive_m = run_events(
            ep.ball_states,
            ep.target_samples,
            fps=fps,
            locked_at_frame=ep.locked_at_frame,
        )
        all_events.extend(events)
        drive_contact_m += epoch_drive_m
        aggregated = PlayerEventCounts(
            Pass=aggregated.Pass + counts.Pass,
            Shot=aggregated.Shot + counts.Shot,
            Goal=aggregated.Goal + counts.Goal,
            Drive=aggregated.Drive + counts.Drive,
        )
        counted += 1
        visible, post = _epoch_quality(ep)
        score = (visible, post)
        if score > best_score:
            best_score = score
            best = ep
    logger.info(
        "[BallEvents] Multi-epoch — epochs=%s counted=%s best=%s aggregated=%s",
        len(epochs),
        counted,
        (best.start_frame, best.end_frame, best.locked_at_frame) if best else None,
        aggregated.model_dump(),
    )
    return aggregated, total_samples, best, all_events, drive_contact_m


def _count_detections(ball_states: list[BallState]) -> int:
    return sum(1 for b in ball_states if b.conf > 0 and not b.predicted)


def _log_event_summary(
    *,
    timeline: list,
    possession: list,
    locked_at_frame: int,
    counts: PlayerEventCounts,
    ball_samples: int,
) -> None:
    import json
    import time

    from backend.app.pipeline.ball_events.possession import PossessionState
    from backend.app.pipeline.ball_events.timeline import dist_m, dist_px

    path = os.environ.get("BALL_EVENTS_SUMMARY_LOG", "").strip()
    if not path:
        return
    post = [
        e
        for e in timeline
        if has_target_identification(e.target, locked_at_frame)[0]
    ]
    possessed = sum(
        1
        for e, p in zip(timeline, possession, strict=True)
        if has_target_identification(e.target, locked_at_frame)[0]
        and p.state
        in (PossessionState.TARGET_POSSESSED, PossessionState.TARGET_POSSESSED_GAP)
    )
    px_dists: list[float] = []
    m_dists: list[float] = []
    for e in post:
        t = e.target
        b = e.ball
        if t.target_px is not None and b.cx_px > 0 and b.cy_px > 0:
            px_dists.append(dist_px(t.target_px, b.cx_px, b.cy_px))
        if t.target_m is not None and b.smooth_m is not None:
            m_dists.append(dist_m(t.target_m, b.smooth_m))
    possess_px = _env_float("POSSESS_RADIUS_PX", 90.0)
    payload = {
        "sessionId": "6b7c41",
        "runId": "summary",
        "hypothesisId": "H6-anchor",
        "location": "run_events.py:_log_event_summary",
        "message": "ball event pipeline summary",
        "data": {
            "post_lock_frames": len(post),
            "possessed_frames": possessed,
            "counts": counts.model_dump(),
            "ball_samples": ball_samples,
            "target_visible_frames": sum(1 for e in post if e.target.target_visible),
            "bbox_track_frames": sum(
                1 for e in post if e.target.segment_source == "bbox_track"
            ),
            "frames_with_target_px": sum(1 for e in post if e.target.target_px),
            "min_px_dist": min(px_dists) if px_dists else None,
            "min_m_dist": min(m_dists) if m_dists else None,
            "px_dist_samples": len(px_dists),
            "frames_within_possess_px": sum(1 for d in px_dists if d < possess_px),
            "proximity_m_count": len(m_dists),
            "proximity_px_count": len(px_dists),
        },
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except OSError:
        pass


def _write_diagnostic(
    *,
    debug_frames: list[FrameDiag],
    total_frames: int,
    fps: float,
    locked_at_frame: int,
    possess_radius_m: float,
    gap_possess_radius_m: float,
    gap_max_frames: int,
    pass_min_speed_m_s: float,
    shot_min_speed_m_s: float,
    ball_samples: int,
    events_found: dict[str, int],
) -> None:
    path = f"/tmp/ball_events_diag_{int(time.time())}.json"
    payload = {
        "meta": {
            "total_frames": total_frames,
            "fps": fps,
            "locked_at_frame": locked_at_frame,
            "possess_radius_m": possess_radius_m,
            "gap_possess_radius_m": gap_possess_radius_m,
            "gap_max_frames": gap_max_frames,
            "pass_min_speed_m_s": pass_min_speed_m_s,
            "shot_min_speed_m_s": shot_min_speed_m_s,
            "ball_samples": ball_samples,
            "events_found": events_found,
        },
        "frames": [asdict(f) for f in debug_frames],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)
    logger.info("[BallEvents] Diagnostic written to %s", path)
