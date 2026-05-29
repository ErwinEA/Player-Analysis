"""Possession FSM unit tests."""

from __future__ import annotations

from backend.app.pipeline.ball_events.ball_detector import BallState
from backend.app.pipeline.ball_events.possession import PossessionState, PossessionTracker
from backend.app.pipeline.ball_events.timeline import FrameTimeline, TargetFrameSample


def _ball(frame: int, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> BallState:
    return BallState(
        frame=frame,
        cx_px=100.0,
        cy_px=100.0,
        conf=0.9,
        pitch_m=(x, y),
        smooth_m=(x, y),
        vel_m_per_frame=(vx, vy),
        predicted=False,
    )


def _entry(
    frame: int,
    ball: BallState,
    target_m: tuple[float, float] | None,
    visible: bool,
    *,
    target_px: tuple[float, float] | None = (100.0, 100.0),
) -> FrameTimeline:
    return FrameTimeline(
        frame=frame,
        ball=ball,
        target=TargetFrameSample(
            frame=frame,
            target_m=target_m,
            target_visible=visible,
            target_px=target_px,
        ),
    )


def test_target_possessed_when_close():
    tracker = PossessionTracker(fps=30.0)
    e = _entry(0, _ball(0, 50.0, 34.0), (50.2, 34.1), True, target_px=(102.0, 101.0))
    pf = tracker.update(e)
    assert pf.state == PossessionState.TARGET_POSSESSED
    assert pf.confidence == "high"


def test_gap_sticky_possession():
    tracker = PossessionTracker(fps=30.0)
    e0 = _entry(0, _ball(0, 50.0, 34.0), (50.0, 34.0), True)
    tracker.update(e0)
    e1 = _entry(1, _ball(1, 50.1, 34.0), (50.1, 34.0), False)
    pf = tracker.update(e1)
    assert pf.state == PossessionState.TARGET_POSSESSED_GAP
    assert pf.confidence == "medium"
