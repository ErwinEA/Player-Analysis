"""Event detection on synthetic timelines."""

from __future__ import annotations

from backend.app.pipeline.ball_events.ball_detector import BallState
from backend.app.pipeline.ball_events.events import (
    aggregate_counts,
    compute_drive_contact_m,
    detect_events,
)
from backend.app.pipeline.ball_events.possession import PossessionTracker
from backend.app.pipeline.ball_events.timeline import FrameTimeline, TargetFrameSample


def _timeline_pass_release(fps: float = 30.0) -> tuple[list[FrameTimeline], list]:
    timeline: list[FrameTimeline] = []
    for f in range(40):
        if f < 20:
            ball = BallState(
                frame=f,
                cx_px=0,
                cy_px=0,
                conf=0.9,
                pitch_m=(50.0, 34.0),
                smooth_m=(50.0, 34.0),
                vel_m_per_frame=(0.0, 0.0),
                predicted=False,
            )
            target = TargetFrameSample(
                frame=f, target_m=(50.0, 34.0), target_visible=True
            )
        else:
            ball = BallState(
                frame=f,
                cx_px=0,
                cy_px=0,
                conf=0.0,
                pitch_m=None,
                smooth_m=(50.0 + (f - 20) * 0.5, 34.0),
                vel_m_per_frame=(0.0, 0.2),
                predicted=f > 20,
            )
            target = TargetFrameSample(
                frame=f, target_m=(50.0, 34.0), target_visible=False
            )
        timeline.append(FrameTimeline(frame=f, ball=ball, target=target))

    tracker = PossessionTracker(fps=fps)
    possession = [tracker.update(e) for e in timeline]
    return timeline, possession


def test_pass_detected_on_release():
    timeline, possession = _timeline_pass_release()
    events = detect_events(timeline, possession, fps=30.0, locked_at_frame=0)
    kinds = [e.kind for e in events]
    assert "Pass" in kinds


def test_aggregate_counts():
    from backend.app.pipeline.ball_events.events import DetectedEvent

    counts = aggregate_counts(
        [
            DetectedEvent(frame=1, kind="Pass", confidence="high"),
            DetectedEvent(frame=2, kind="Pass", confidence="high"),
        ]
    )
    assert counts.Pass == 2
    assert counts.Goal == 0


def _timeline_long_drive(*, fps: float = 25.0, drive_frames: int = 30) -> tuple[list, list]:
    """Target dribbles in a straight line, then releases."""
    timeline: list[FrameTimeline] = []
    release_at = drive_frames
    total = release_at + 5
    for f in range(total):
        x = 50.0 + f * 0.5
        if f < release_at:
            ball = BallState(
                frame=f,
                cx_px=0,
                cy_px=0,
                conf=0.9,
                pitch_m=(x, 34.0),
                smooth_m=(x, 34.0),
                vel_m_per_frame=(0.0, 0.0),
                predicted=False,
            )
            target = TargetFrameSample(
                frame=f,
                target_m=(x, 34.0),
                target_visible=True,
                identification_type="locked",
            )
        else:
            ball = BallState(
                frame=f,
                cx_px=0,
                cy_px=0,
                conf=0.0,
                pitch_m=None,
                smooth_m=(x + (f - release_at) * 0.8, 34.0),
                vel_m_per_frame=(0.8, 0.0),
                predicted=True,
            )
            target = TargetFrameSample(
                frame=f,
                target_m=(50.0 + (release_at - 1) * 0.5, 34.0),
                target_visible=False,
                identification_type="locked",
            )
        timeline.append(FrameTimeline(frame=f, ball=ball, target=target))

    tracker = PossessionTracker(fps=fps)
    possession = [tracker.update(e) for e in timeline]
    return timeline, possession


def test_drive_detected_on_long_possession(monkeypatch):
    monkeypatch.setenv("DRIVE_MIN_FRAMES", "15")
    timeline, possession = _timeline_long_drive(drive_frames=30)
    events = detect_events(timeline, possession, fps=25.0, locked_at_frame=0)
    kinds = [e.kind for e in events]
    assert "Drive" in kinds


def test_drive_survives_identification_gap(monkeypatch):
    monkeypatch.setenv("DRIVE_MIN_FRAMES", "15")
    timeline, possession = _timeline_long_drive(drive_frames=30)
    for entry in timeline[10:15]:
        entry.target.identification_type = None
    events = detect_events(timeline, possession, fps=25.0, locked_at_frame=0)
    assert any(e.kind == "Drive" for e in events)


def test_drive_detected_when_possession_ends_at_video_end(monkeypatch):
    monkeypatch.setenv("DRIVE_MIN_FRAMES", "15")
    timeline: list[FrameTimeline] = []
    for f in range(30):
        x = 50.0 + f * 0.5
        ball = BallState(
            frame=f,
            cx_px=0,
            cy_px=0,
            conf=0.9,
            pitch_m=(x, 34.0),
            smooth_m=(x, 34.0),
            vel_m_per_frame=(0.0, 0.0),
            predicted=False,
        )
        target = TargetFrameSample(
            frame=f,
            target_m=(x, 34.0),
            target_visible=True,
            identification_type="locked",
        )
        timeline.append(FrameTimeline(frame=f, ball=ball, target=target))
    tracker = PossessionTracker(fps=25.0)
    possession = [tracker.update(e) for e in timeline]
    events = detect_events(timeline, possession, fps=25.0, locked_at_frame=0)
    assert any(e.kind == "Drive" for e in events)


def test_compute_drive_contact_m():
    timeline, possession = _timeline_long_drive(drive_frames=30)
    contact_m = compute_drive_contact_m(
        timeline, possession, locked_at_frame=0
    )
    # 29 steps × 0.5 m per frame along x ≈ 14.5 m
    assert contact_m > 10.0


def _timeline_shot_release(*, fps: float = 30.0, release_at: int = 20) -> tuple[list, list]:
    timeline: list[FrameTimeline] = []
    for f in range(50):
        if f < release_at:
            ball = BallState(
                frame=f,
                cx_px=0,
                cy_px=0,
                conf=0.9,
                pitch_m=(50.0, 34.0),
                smooth_m=(50.0, 34.0),
                vel_m_per_frame=(0.0, 0.0),
                predicted=False,
            )
            target = TargetFrameSample(
                frame=f,
                target_m=(50.0, 34.0),
                target_visible=True,
                identification_type="ocr",
            )
        elif f < release_at + 5:
            ball = BallState(
                frame=f,
                cx_px=0,
                cy_px=0,
                conf=0.0,
                pitch_m=None,
                smooth_m=(50.0 + max(0, f - release_at) * 0.5, 34.0),
                vel_m_per_frame=(0.5, 0.0),
                predicted=True,
            )
            target = TargetFrameSample(
                frame=f,
                target_m=(50.0, 34.0),
                target_visible=False,
                identification_type=None,
            )
        else:
            ball = BallState(
                frame=f,
                cx_px=0,
                cy_px=0,
                conf=0.0,
                pitch_m=None,
                smooth_m=(50.0 + (f - release_at) * 0.5, 34.0),
                vel_m_per_frame=(0.1, 0.0),
                predicted=True,
            )
            target = TargetFrameSample(
                frame=f,
                target_m=(50.0, 34.0),
                target_visible=False,
                identification_type="ocr",
            )
        timeline.append(FrameTimeline(frame=f, ball=ball, target=target))

    tracker = PossessionTracker(fps=fps)
    possession = [tracker.update(e) for e in timeline]
    return timeline, possession


def test_shot_survives_identification_gap():
    """Release speed must be read at the gap frame, not when OCR resumes."""
    timeline, possession = _timeline_shot_release()
    events = detect_events(timeline, possession, fps=30.0, locked_at_frame=None)
    assert any(e.kind == "Shot" for e in events)
    shot = next(e for e in events if e.kind == "Shot")
    assert shot.frame == 20
