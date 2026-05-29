"""Tests for resolve_target_bbox candidate validity gating (Fix #1)."""

from __future__ import annotations

import numpy as np

from backend.app.pipeline.pitch_homography import build_calibration
from backend.app.pipeline.segmentation import (
    is_valid_reid_candidate_bbox,
    resolve_target_bbox,
)

FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)
PROTO = np.array([1.0, 0.0, 0.0], dtype=np.float32)


class _StubExtractor:
    """Returns the prototype for every crop so similarity is 1.0 for any candidate."""

    def embed(self, crop):  # noqa: ANN001
        return PROTO


def _cal():
    return build_calibration(
        "t",
        [[115, 425], [905, 495], [1265, 705], [35, 705]],
        image_size=(1280, 720),
    )


def test_picks_player_over_scoreboard_when_lock_missing():
    # Scoreboard banner (wide) has the same appearance score as the player, but the
    # shape gate must drop it so the upright player box is chosen.
    tracks = [
        {"track_id": 1, "bbox": [100.0, 40.0, 700.0, 110.0]},  # wide banner
        {"track_id": 2, "bbox": [500.0, 300.0, 560.0, 470.0]},  # upright player
    ]
    result = resolve_target_bbox(
        tracks,
        lock_track_id=99,  # not present -> ReID fallback path
        reid_prototype=PROTO,
        reid_extractor=_StubExtractor(),
        frame=FRAME,
    )
    assert result is not None
    assert result[0] == 2
    assert result[2] is True


def test_returns_none_when_only_candidate_is_off_pitch():
    cal = _cal()
    # Upright player shape, but feet near the top of the frame project off-pitch.
    tracks = [{"track_id": 5, "bbox": [600.0, 5.0, 660.0, 120.0]}]
    result = resolve_target_bbox(
        tracks,
        lock_track_id=99,
        reid_prototype=PROTO,
        reid_extractor=_StubExtractor(),
        frame=FRAME,
        calibration=cal,
    )
    assert result is None


def test_locked_track_with_invalid_box_falls_through():
    # The locked id resolves onto a wide banner; without ReID it must not be trusted.
    tracks = [{"track_id": 7, "bbox": [100.0, 40.0, 900.0, 110.0]}]
    result = resolve_target_bbox(
        tracks,
        lock_track_id=7,
        reid_prototype=None,
        reid_extractor=None,
        frame=FRAME,
    )
    assert result is None


def test_locked_track_with_valid_box_returned():
    tracks = [{"track_id": 7, "bbox": [500.0, 300.0, 560.0, 470.0]}]
    result = resolve_target_bbox(
        tracks,
        lock_track_id=7,
        reid_prototype=None,
        reid_extractor=None,
        frame=FRAME,
    )
    assert result == (7, [500.0, 300.0, 560.0, 470.0], False)


def test_validity_helper_on_pitch_gate():
    cal = _cal()
    # A plausible player on the pitch passes both gates.
    assert is_valid_reid_candidate_bbox([600.0, 560.0, 650.0, 700.0], (720, 1280), calibration=cal)
