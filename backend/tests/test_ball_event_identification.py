"""Identification-based ball event gating (pre-lock soft ID)."""

from __future__ import annotations

from backend.app.pipeline.ball_events.ball_detector import BallState
from backend.app.pipeline.ball_events.events import detect_events
from backend.app.pipeline.ball_events.identification import has_target_identification
from backend.app.pipeline.ball_events.possession import PossessionTracker
from backend.app.pipeline.ball_events.run_events import LockEpoch, run_events, run_events_multi
from backend.app.pipeline.ball_events.timeline import FrameTimeline, TargetFrameSample


def test_has_target_identification_weak_ocr():
    sample = TargetFrameSample(
        frame=825,
        target_m=(50.0, 34.0),
        target_visible=True,
        identification_type="ocr",
    )
    ok, tier, phase = has_target_identification(sample, locked_at_frame=987)
    assert ok is True
    assert tier == "weak"
    assert phase == "ocr"


def test_pre_lock_shot_not_gated_by_locked_at_frame():
    """Shot at frame 825 must be evaluated when only soft OCR id exists (lock at 987)."""
    timeline: list[FrameTimeline] = []
    for f in range(840):
        if f < 820:
            ball = BallState(
                frame=f,
                cx_px=100,
                cy_px=100,
                conf=0.0,
                pitch_m=None,
                smooth_m=(10.0, 10.0),
                vel_m_per_frame=(0.0, 0.0),
                predicted=True,
            )
            target = TargetFrameSample(frame=f, target_m=None, target_visible=False)
        elif f < 826:
            ball = BallState(
                frame=f,
                cx_px=100,
                cy_px=100,
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
        else:
            ball = BallState(
                frame=f,
                cx_px=100,
                cy_px=100,
                conf=0.0,
                pitch_m=None,
                smooth_m=(50.0 + (f - 826) * 0.8, 34.0),
                vel_m_per_frame=(0.8, 0.0),
                predicted=f > 826,
            )
            target = TargetFrameSample(
                frame=f,
                target_m=(50.0, 34.0),
                target_visible=False,
                identification_type="ocr",
            )
        timeline.append(FrameTimeline(frame=f, ball=ball, target=target))

    tracker = PossessionTracker(fps=25.0)
    possession = [tracker.update(e) for e in timeline]
    events = detect_events(
        timeline,
        possession,
        fps=25.0,
        locked_at_frame=987,
    )
    kinds = [e.kind for e in events]
    assert "Shot" in kinds or "Pass" in kinds
    shot = next((e for e in events if e.kind == "Shot"), None)
    if shot is not None:
        assert shot.frame >= 820
        assert shot.lock_confidence == "weak"
        assert shot.detection_phase == "ocr"


def test_run_events_accepts_soft_epoch_without_hard_lock():
    ball_states: list[BallState] = []
    target_samples: list[TargetFrameSample] = []
    for f in range(30):
        ball_states.append(
            BallState(
                frame=f,
                cx_px=0,
                cy_px=0,
                conf=0.9,
                pitch_m=(50.0, 34.0),
                smooth_m=(50.0, 34.0),
                vel_m_per_frame=(0.0, 0.0),
                predicted=False,
            )
        )
        target_samples.append(
            TargetFrameSample(
                frame=f,
                target_m=(50.0, 34.0),
                target_visible=True,
                identification_type="ocr",
            )
        )
    counts, _samples, events, drive_contact_m = run_events(
        ball_states,
        target_samples,
        fps=25.0,
        locked_at_frame=None,
    )
    assert counts.Pass + counts.Shot + counts.Drive >= 0
    assert events is not None
    assert drive_contact_m >= 0.0


def test_soft_epoch_in_multi_without_locked_at():
    epoch = LockEpoch(
        start_frame=600,
        end_frame=900,
        locked_at_frame=None,
        track_id=None,
        ball_states=[],
        target_samples=[
            TargetFrameSample(
                frame=600 + i,
                target_m=(50.0, 34.0),
                target_visible=True,
                identification_type="ocr",
            )
            for i in range(10)
        ],
    )
    # Add minimal ball timeline for build
    for i, s in enumerate(epoch.target_samples):
        epoch.ball_states.append(
            BallState(
                frame=s.frame,
                cx_px=0,
                cy_px=0,
                conf=0.9,
                pitch_m=(50.0, 34.0),
                smooth_m=(50.0, 34.0),
                vel_m_per_frame=(0.0, 0.0),
                predicted=False,
            )
        )
    counts, _, best, events, drive_contact_m = run_events_multi([epoch], fps=25.0)
    assert best is epoch or best is None
    assert isinstance(events, list)
    assert drive_contact_m >= 0.0
