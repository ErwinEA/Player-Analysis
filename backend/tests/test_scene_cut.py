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
