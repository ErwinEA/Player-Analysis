"""Event detection on synthetic timelines."""

from __future__ import annotations

from backend.app.pipeline.ball_events.ball_detector import BallState
from backend.app.pipeline.ball_events.events import aggregate_counts, detect_events
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
