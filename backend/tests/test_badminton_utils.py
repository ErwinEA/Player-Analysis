"""Tests for badminton disambiguation helpers."""

from backend.app.pipeline.pitch_homography import build_calibration
from backend.app.pipeline.sports.badminton_utils import (
    centroid_y,
    disambiguate_by_court_side,
    disambiguate_with_fallback,
    net_line_y_px,
)


def _cal(w: int = 1280, h: int = 720):
    corners = [
        [w * 0.2, h * 0.2],
        [w * 0.8, h * 0.2],
        [w * 0.8, h * 0.8],
        [w * 0.2, h * 0.8],
    ]
    return build_calibration(
        "test",
        image_corners=corners,
        image_size=(w, h),
        length_m=13.4,
        width_m=6.1,
    )


def test_centroid_y():
    assert centroid_y([0.0, 100.0, 50.0, 200.0]) == 150.0


def test_net_line_y_px_near_mid_frame():
    cal = _cal()
    net_y = net_line_y_px(cal)
    assert 0 < net_y < 720


def test_disambiguate_near_court():
    net_y = 360.0
    tracks = [
        {"track_id": 1, "bbox": [100.0, 400.0, 200.0, 500.0]},
        {"track_id": 2, "bbox": [100.0, 100.0, 200.0, 200.0]},
    ]
    assert disambiguate_by_court_side(tracks, "near", net_y) == 1
    assert disambiguate_by_court_side(tracks, "far", net_y) == 2


def test_disambiguate_ambiguous_when_both_same_side():
    net_y = 360.0
    tracks = [
        {"track_id": 1, "bbox": [100.0, 400.0, 200.0, 500.0]},
        {"track_id": 2, "bbox": [300.0, 420.0, 400.0, 520.0]},
    ]
    assert disambiguate_by_court_side(tracks, "near", net_y) is None


def test_disambiguate_with_color_fallback():
    net_y = 360.0
    tracks = [
        {"track_id": 1, "bbox": [100.0, 400.0, 200.0, 500.0]},
        {"track_id": 2, "bbox": [300.0, 420.0, 400.0, 520.0]},
    ]
    tid, method = disambiguate_with_fallback(
        tracks, "near", net_y, {2: [0.8, 0.9]}
    )
    assert tid == 2
    assert method == "color"
