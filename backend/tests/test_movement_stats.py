"""Tests for movement statistics."""

from __future__ import annotations

from backend.app.pipeline.movement_stats import (
    MovementStatsConfig,
    _filter_trajectory_to_pitch,
    compute_movement_stats,
    compute_movement_stats_from_rows,
    foot_position_pixels,
    trajectory_points_from_rows,
)
from backend.app.pipeline.pitch_homography import (
    PitchCalibration,
    build_calibration,
    is_on_pitch,
)
from backend.app.schemas import Row


def _on_pitch_sprint_rows_fixture() -> tuple[PitchCalibration, list[Row]]:
    """Rows whose feet map inside the calibrated pitch (not frame corners)."""
    cal = build_calibration(
        "test",
        [[115, 425], [905, 495], [1265, 705], [35, 705]],
        image_size=(1280, 720),
    )
    fw, fh = cal.image_size
    foot_ys_px = [550.0, 600.0, 650.0]
    rows: list[Row] = []
    for i, foot_y_px in enumerate(foot_ys_px):
        bbox = [550.0, foot_y_px - 80.0, 650.0, foot_y_px]
        foot_x_px, _ = foot_position_pixels(bbox)
        rows.append(
            Row(
                t=float(i),
                frame=i * 30,
                track=1,
                player=10,
                x=0.5,
                y=0.5,
                bbox=bbox,
                foot_x=foot_x_px / fw,
                foot_y=foot_y_px / fh,
            )
        )
    return cal, rows


def _assert_all_feet_on_pitch(cal: PitchCalibration, rows: list[Row]) -> None:
    points = trajectory_points_from_rows(rows, fps=30.0, calibration=cal)
    assert len(points) == len(rows), "expected one trajectory point per row"
    length_m = cal.pitch_length_m
    width_m = cal.pitch_width_m
    for _t, x_m, y_m in points:
        assert is_on_pitch(x_m, y_m, length_m=length_m, width_m=width_m), (
            f"foot maps off pitch: ({x_m:.2f}, {y_m:.2f})"
        )
    filtered = _filter_trajectory_to_pitch(points, cal)
    assert len(filtered) == len(points), "on-pitch filter dropped fixture feet"


def test_constant_speed_metrics():
    points = [(float(i), 0.0, float(i) * 5.0) for i in range(6)]
    stats = compute_movement_stats(points)
    assert stats.distance_m == 25.0
    assert stats.max_speed_m_s == 5.0
    assert stats.avg_speed_m_s == 5.0


def test_movement_from_rows_on_pitch():
    cal, rows = _on_pitch_sprint_rows_fixture()
    _assert_all_feet_on_pitch(cal, rows)
    stats = compute_movement_stats_from_rows(rows, fps=30.0, calibration=cal)
    assert stats.units == "meters"
    assert stats.sample_count == 3
    assert stats.distance_m > 0


def test_sprint_with_row_models():
    cal, rows = _on_pitch_sprint_rows_fixture()
    stats = compute_movement_stats_from_rows(
        rows,
        fps=30.0,
        calibration=cal,
        config=MovementStatsConfig(sprint_speed_m_s=0.5, min_sprint_duration_s=0.5),
    )
    assert stats.units == "meters"
    assert stats.sample_count == 3
    assert stats.distance_m > 0
    assert stats.sprint_count >= 1
