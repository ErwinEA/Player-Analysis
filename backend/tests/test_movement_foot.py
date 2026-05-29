"""Foot fields on Row drive trajectory when normalized."""

from __future__ import annotations

from backend.app.pipeline.movement_stats import foot_position_pixels, trajectory_points_from_rows
from backend.app.pipeline.pitch_homography import build_calibration
from backend.app.schemas import Row


def test_trajectory_uses_normalized_foot_when_set():
    cal = build_calibration(
        "t",
        [[115, 425], [905, 495], [1265, 705], [35, 705]],
        image_size=(1280, 720),
    )
    fw, fh = cal.image_size
    bbox = [100.0, 50.0, 200.0, 400.0]
    foot_px = foot_position_pixels(bbox)
    row_foot = Row(
        t=0.0,
        frame=0,
        track=1,
        player=10,
        x=0.5,
        y=0.5,
        bbox=bbox,
        foot_x=(foot_px[0] + 40.0) / fw,
        foot_y=(foot_px[1] - 10.0) / fh,
    )
    row_bbox = Row(
        t=0.0,
        frame=0,
        track=1,
        player=10,
        x=0.5,
        y=0.5,
        bbox=bbox,
    )
    p_foot = trajectory_points_from_rows([row_foot], fps=30.0, calibration=cal)[0]
    p_bbox = trajectory_points_from_rows([row_bbox], fps=30.0, calibration=cal)[0]
    assert p_foot != p_bbox
