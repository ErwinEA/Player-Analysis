"""Multi-epoch aggregation and best-epoch selection."""

from __future__ import annotations

from backend.app.pipeline.ball_events.ball_detector import BallState
from backend.app.pipeline.ball_events.run_events import LockEpoch, run_events_multi
from backend.app.pipeline.ball_events.timeline import TargetFrameSample


def _pass_epoch(start_frame: int, *, on_pitch: bool) -> LockEpoch:
    """A 40-frame lock window: possession then a fast release (a Pass)."""
    x, y = (50.0, 34.0) if on_pitch else (300.0, 300.0)
    ball_states: list[BallState] = []
    target_samples: list[TargetFrameSample] = []
    for i in range(40):
        f = start_frame + i
        if i < 20:
            ball_states.append(
                BallState(
                    frame=f,
                    cx_px=0,
                    cy_px=0,
                    conf=0.9,
                    pitch_m=(x, y),
                    smooth_m=(x, y),
                    vel_m_per_frame=(0.0, 0.0),
                    predicted=False,
                )
            )
            target_samples.append(
                TargetFrameSample(
                    frame=f,
                    target_m=(x, y),
                    target_visible=True,
                    identification_type="locked",
                )
            )
        else:
            ball_states.append(
                BallState(
                    frame=f,
                    cx_px=0,
                    cy_px=0,
                    conf=0.0,
                    pitch_m=None,
                    smooth_m=(x + (i - 20) * 0.5, y),
                    vel_m_per_frame=(0.0, 0.2),
                    predicted=i > 20,
                )
            )
            target_samples.append(
                TargetFrameSample(frame=f, target_m=(x, y), target_visible=False)
            )
    return LockEpoch(
        start_frame=start_frame,
        end_frame=start_frame + 40,
        locked_at_frame=start_frame,
        track_id=start_frame + 1,
        ball_states=ball_states,
        target_samples=target_samples,
    )


def test_aggregates_passes_across_good_epochs():
    epochs = [_pass_epoch(0, on_pitch=True), _pass_epoch(100, on_pitch=True)]
    counts, _samples, best, _events, _drive_m = run_events_multi(epochs, fps=30.0)
    assert counts.Pass == 2
    assert best is not None


def test_off_pitch_epoch_contributes_nothing():
    good = _pass_epoch(0, on_pitch=True)
    bad = _pass_epoch(100, on_pitch=False)
    counts, _samples, best, _events, _drive_m = run_events_multi([good, bad], fps=30.0)
    assert counts.Pass == 1
    assert best is good


def test_no_lock_epoch_skipped():
    good = _pass_epoch(0, on_pitch=True)
    no_lock = LockEpoch(
        start_frame=100,
        end_frame=140,
        locked_at_frame=None,
        track_id=None,
        ball_states=[],
        target_samples=[],
    )
    counts, _samples, best, _events, _drive_m = run_events_multi([good, no_lock], fps=30.0)
    assert counts.Pass == 1
    assert best is good


def test_best_epoch_prefers_more_visible_frames():
    short = _pass_epoch(0, on_pitch=True)
    short.target_samples = short.target_samples[:10]
    short.ball_states = short.ball_states[:10]
    long = _pass_epoch(100, on_pitch=True)
    _counts, _samples, best, _events, _drive_m = run_events_multi([short, long], fps=30.0)
    assert best is long
