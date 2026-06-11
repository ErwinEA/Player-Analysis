"""Tests for the broadcast scene-cut detector."""

from __future__ import annotations

import numpy as np

from backend.app.pipeline.scene_cut import SceneCutDetector


def _solid(value: int) -> np.ndarray:
    return np.full((64, 64, 3), value, dtype=np.uint8)


def test_first_frame_is_not_a_cut():
    det = SceneCutDetector(threshold=0.65, suppress_frames=5)
    assert det.update(_solid(0)) is False


def test_hard_cut_detected_and_suppression_window():
    det = SceneCutDetector(threshold=0.65, suppress_frames=3)
    det.update(_solid(0))            # establish baseline (black)
    assert det.update(_solid(255)) is True  # black -> white is a hard cut
    assert det.is_suppressed is True

    # Same frame repeated: no new cut, suppression counts down.
    det.update(_solid(255))
    det.update(_solid(255))
    det.update(_solid(255))
    assert det.is_suppressed is False


def test_no_cut_between_similar_frames():
    det = SceneCutDetector(threshold=0.65, suppress_frames=3)
    det.update(_solid(120))
    assert det.update(_solid(122)) is False


def _shuttle_overlay_px(
    *,
    suppressed: bool,
    det_px: tuple[float, float] | None,
    kalman,
    stride_skip: bool,
) -> tuple[float, float] | None:
    """Mirror run.py shuttle overlay assignment (detection / predict / stride)."""
    if not stride_skip:
        if det_px is not None:
            smooth = kalman.update(det_px[0], det_px[1])
            return det_px if suppressed else smooth
        if not suppressed:
            return kalman.predict()
        return None
    if not suppressed:
        return kalman.predict()
    return None


def test_shuttle_overlay_skips_kalman_predict_during_suppression():
    """run.py must not coast Kalman into overlay when suppressed, even if seeded."""
    from backend.app.pipeline.badminton.shuttle_detector import ShuttleKalman

    scene = SceneCutDetector(threshold=0.65, suppress_frames=5)
    kalman = ShuttleKalman(gap_max=15)

    scene.update(_solid(0))
    kalman.update(100.0, 200.0)
    assert scene.update(_solid(255)) is True
    kalman.reset()

    # No detection on cut frame — overlay stays None (no predict branch).
    assert scene.is_suppressed is True
    assert (
        _shuttle_overlay_px(
            suppressed=scene.is_suppressed,
            det_px=None,
            kalman=kalman,
            stride_skip=False,
        )
        is None
    )

    # Raw detection seeds Kalman during suppression; next gap must not coast.
    scene.update(_solid(255))
    raw = (50.0, 60.0)
    overlay = _shuttle_overlay_px(
        suppressed=scene.is_suppressed,
        det_px=raw,
        kalman=kalman,
        stride_skip=False,
    )
    assert overlay == raw
    assert (
        _shuttle_overlay_px(
            suppressed=scene.is_suppressed,
            det_px=None,
            kalman=kalman,
            stride_skip=False,
        )
        is None
    )

    # After suppression lifts, predict/coast resumes.
    for _ in range(5):
        scene.update(_solid(255))
    assert scene.is_suppressed is False
    assert (
        _shuttle_overlay_px(
            suppressed=scene.is_suppressed,
            det_px=None,
            kalman=kalman,
            stride_skip=False,
        )
        is not None
    )
