"""Tests for badminton disambiguation helpers."""

from backend.app.pipeline.pitch_homography import build_calibration
from backend.app.pipeline.sports.badminton_utils import (
    centroid_y,
    court_side_to_metre_half,
    disambiguate_by_court_side,
    disambiguate_with_fallback,
    dominant_court_half,
    filter_positions_to_court_half,
    filter_rows_by_court_side,
    mask_grid_to_court_half,
    metre_half_for_court_side,
    net_line_x_m,
    net_line_y_px,
    position_in_court_half,
    row_on_court_side,
)
from backend.app.schemas import Row


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
    color_scores = {2: [0.8, 0.85, 0.9, 0.88, 0.87]}
    tid, method = disambiguate_with_fallback(
        tracks, "near", net_y, color_scores
    )
    assert tid == 2
    assert method == "color"


def test_apply_color_lock_sets_method():
    from backend.app.pipeline.matcher import TrackIdentityMatcher

    matcher = TrackIdentityMatcher(target_jersey=0, target_name="")
    lock = matcher.apply_color_lock(7, confidence=0.42)
    assert lock.method == "color"
    assert matcher.lock is not None
    assert matcher.lock.method == "color"


def test_row_on_court_side_uses_foot_y():
    net_y = 360.0
    near_row = Row(
        t=0.0,
        frame=0,
        track=1,
        player=None,
        x=0.5,
        y=0.7,
        bbox=[100.0, 400.0, 200.0, 500.0],
        foot_y=0.75,
    )
    far_row = Row(
        t=0.0,
        frame=1,
        track=2,
        player=None,
        x=0.5,
        y=0.2,
        bbox=[100.0, 100.0, 200.0, 200.0],
        foot_y=0.15,
    )
    assert row_on_court_side(near_row, "near", net_y, image_height=720)
    assert not row_on_court_side(far_row, "near", net_y, image_height=720)
    assert row_on_court_side(far_row, "far", net_y, image_height=720)


def test_filter_rows_by_court_side_drops_opponent():
    cal = _cal()
    net_y = net_line_y_px(cal)
    rows = [
        Row(
            t=0.0,
            frame=0,
            track=1,
            player=None,
            x=0.5,
            y=0.7,
            bbox=[100.0, 400.0, 200.0, 500.0],
            foot_y=(net_y + 50.0) / 720.0,
        ),
        Row(
            t=0.1,
            frame=1,
            track=2,
            player=None,
            x=0.5,
            y=0.2,
            bbox=[100.0, 100.0, 200.0, 200.0],
            foot_y=(net_y - 50.0) / 720.0,
        ),
    ]
    near_only = filter_rows_by_court_side(rows, cal, "near")
    assert len(near_only) == 1
    assert near_only[0].track == 1


def test_position_in_court_half():
    cal = _cal()
    mid = net_line_x_m(cal)
    assert position_in_court_half(mid * 0.25, cal, "low")
    assert not position_in_court_half(mid * 1.75, cal, "low")
    assert position_in_court_half(mid * 1.75, cal, "high")


def test_court_side_to_metre_half_returns_valid_half():
    cal = _cal()
    assert court_side_to_metre_half(cal, "near") in ("low", "high")
    assert court_side_to_metre_half(cal, "far") in ("low", "high")


def test_court_side_to_metre_half_uses_image_probe_x():
    cal = _cal()
    mid = net_line_x_m(cal)
    left = court_side_to_metre_half(cal, "near", image_probe_x=cal.image_size[0] * 0.3)
    right = court_side_to_metre_half(cal, "near", image_probe_x=cal.image_size[0] * 0.7)
    assert left == "low"
    assert right == "high"


def test_metre_half_for_court_side_uses_median_when_enough_samples():
    cal = _cal()
    mid = net_line_x_m(cal)
    positions = [(mid * 0.3, 3.0)] * 2 + [(mid * 1.7, 3.0)] * 8
    half = metre_half_for_court_side(positions, cal, "near")
    assert half == "high"


def test_dominant_court_half_and_filter():
    cal = _cal()
    mid = net_line_x_m(cal)
    positions = [(mid * 0.3, 3.0)] * 8 + [(mid * 1.7, 3.0)] * 2
    assert dominant_court_half(positions, cal) == "low"
    kept = filter_positions_to_court_half(positions, cal, "near")
    near_half = court_side_to_metre_half(cal, "near")
    assert len(kept) >= 2
    for x, _ in kept:
        if near_half == "low":
            assert x < mid
        else:
            assert x > mid


def test_mask_grid_to_court_half_zeros_opponent_columns():
    import numpy as np

    cal = _cal()
    grid = np.ones((34, 53), dtype=np.float32)
    masked = mask_grid_to_court_half(grid, cal, "low")
    mid_col = 53 // 2
    assert masked[:, :mid_col].sum() > 0
    assert masked[:, mid_col + 1 :].sum() == 0.0


def test_disambiguate_single_player_on_court_side():
    net_y = 360.0
    tracks = [{"track_id": 5, "bbox": [100.0, 400.0, 200.0, 500.0]}]
    color_scores = {5: [0.8, 0.85, 0.9, 0.88, 0.87]}
    tid, method = disambiguate_with_fallback(
        tracks, "near", net_y, color_scores, min_tracks=2
    )
    assert tid == 5
    assert method == "court_side_single"


def test_net_line_y_override():
    from dataclasses import replace

    cal = _cal()
    overridden = replace(cal, net_line_y_override=400.0)
    assert net_line_y_px(overridden) == 400.0


def test_color_fallback_rejects_weak_scores():
    net_y = 360.0
    tracks = [
        {"track_id": 1, "bbox": [100.0, 400.0, 200.0, 500.0]},
        {"track_id": 2, "bbox": [300.0, 420.0, 400.0, 520.0]},
    ]
    tid, method = disambiguate_with_fallback(
        tracks,
        "near",
        net_y,
        {2: [0.05, 0.06, 0.04, 0.05, 0.05]},
    )
    assert tid is None
    assert method == "none"
