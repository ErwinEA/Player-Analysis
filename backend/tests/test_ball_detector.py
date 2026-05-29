"""Tests for ball detector and Kalman gap handling."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.pipeline.ball_events.ball_detector import (
    BallDetection,
    BallDetector,
    _PitchKalman,
    resolve_ball_weights,
)
from backend.app.pipeline.pitch_homography import build_calibration


def test_resolve_ball_weights_ignores_invalid_explicit_path():
    # Invalid explicit path falls back to env/default resolution.
    out = resolve_ball_weights("/nonexistent/yolov8n_ball.pt")
    assert out is None or out.endswith("yolov8n_ball.pt")


def test_ball_detector_unavailable_without_weights(monkeypatch):
    monkeypatch.setattr(
        "backend.app.pipeline.ball_events.ball_detector.resolve_ball_weights",
        lambda *_a, **_k: None,
    )
    det = BallDetector()
    assert det.available is False
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    assert det.detect(frame) is None
    assert det.update_pitch(0, None, _dummy_cal()).smooth_m is None


def test_kalman_predict_and_reset():
    kf = _PitchKalman(q=0.5, r=1.0)
    x, y, vx, vy = kf.update(10.0, 20.0)
    assert x == pytest.approx(10.0)
    assert y == pytest.approx(20.0)
    x2, y2, _, _ = kf.predict()
    assert x2 == pytest.approx(10.0 + vx)
    kf.reset()
    assert kf._x is None


def test_update_pitch_gap_resets_after_max(monkeypatch):
    monkeypatch.setattr(
        "backend.app.pipeline.ball_events.ball_detector.resolve_ball_weights",
        lambda *_a, **_k: None,
    )
    det = BallDetector()
    det.available = True
    det._model = object()  # type: ignore[assignment]
    det._gap_max = 2
    cal = _dummy_cal()

    d1 = BallDetection(10, 10, 20, 20, 0.9)
    s1 = det.update_pitch(0, d1, cal)
    assert s1.smooth_m is not None
    assert s1.predicted is False

    s2 = det.update_pitch(1, None, cal)
    assert s2.predicted is True

    s3 = det.update_pitch(2, None, cal)
    assert s3.predicted is True

    s4 = det.update_pitch(3, None, cal)
    assert s4.smooth_m is None


def _dummy_cal():
    w, h = 1280, 720
    corners = [
        [0, 0],
        [w, 0],
        [w, h],
        [0, h],
    ]
    return build_calibration(
        "test",
        image_corners=corners,
        video_path=None,
        frame_index=0,
        image_size=(w, h),
    )
