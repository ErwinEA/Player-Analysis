"""Tests for player position heatmap generation."""

from __future__ import annotations

import numpy as np

from backend.app.pipeline.heatmap import (
    bin_positions,
    build_heatmap_from_rows,
    build_zone_summary,
    meter_positions_from_rows,
    pitch_grid_shape,
    smooth_grid,
    summarize_positions,
)
from backend.app.pipeline.pitch_homography import build_calibration
from backend.app.schemas import Row


def _sample_calibration():
    return build_calibration(
        "test",
        [[115, 425], [905, 495], [1265, 705], [35, 705]],
        image_size=(1280, 720),
    )


def _sample_rows() -> list[Row]:
    rows = []
    for i in range(10):
        x1 = 400.0 + i * 30
        y2 = 500.0 + i * 5
        rows.append(
            Row(
                t=float(i),
                frame=i,
                track=1,
                player=10,
                x=0.5,
                y=0.5,
                bbox=[x1, 400.0, x1 + 40, y2],
            )
        )
    return rows


def test_pitch_grid_shape():
    cell_x, cell_y = pitch_grid_shape(105.0, 68.0, grid_cols=53, grid_rows=34)
    assert abs(cell_x - 105.0 / 53) < 1e-6
    assert abs(cell_y - 68.0 / 34) < 1e-6


def test_bin_positions_accumulates():
    grid = bin_positions(
        [(10.0, 10.0), (10.5, 10.2), (50.0, 30.0)],
        length_m=105.0,
        width_m=68.0,
        grid_cols=53,
        grid_rows=34,
    )
    assert grid.shape == (34, 53)
    assert grid.sum() == 3.0


def test_smooth_grid_preserves_mass():
    grid = np.zeros((10, 10), dtype=np.float32)
    grid[5, 5] = 10.0
    smoothed = smooth_grid(grid, sigma=1.5)
    assert smoothed.sum() > 0
    assert smoothed.max() < grid.max()


def test_meter_positions_filters_out_of_bounds():
    cal = _sample_calibration()
    rows = _sample_rows()
    positions = meter_positions_from_rows(rows, cal)
    for x_m, y_m in positions:
        assert 0 <= x_m <= cal.pitch_length_m
        assert 0 <= y_m <= cal.pitch_width_m


def test_build_heatmap_from_rows():
    cal = _sample_calibration()
    result = build_heatmap_from_rows(_sample_rows(), cal, fps=30.0)
    assert result.sample_count > 0
    assert result.grid_cols == 53
    assert result.grid_rows == 34
    assert len(result.image_png_base64) > 100
    assert result.zone_summary is not None
    zs = result.zone_summary
    assert zs.hottest_x_m is not None
    assert zs.hottest_y_m is not None
    third_total = zs.defensive_third_pct + zs.middle_third_pct + zs.attacking_third_pct
    assert abs(third_total - 100.0) < 0.2
    channel_total = zs.left_pct + zs.center_pct + zs.right_pct
    assert abs(channel_total - 100.0) < 0.2


def test_summarize_positions_empty():
    zs = summarize_positions([], length_m=105.0, width_m=68.0)
    assert zs.defensive_third_pct == 0.0
    assert zs.hottest_x_m is None


def test_summarize_positions_attacking_right():
    # All samples in attacking third, right channel
    positions = [(80.0, 50.0), (85.0, 55.0), (90.0, 60.0)]
    zs = summarize_positions(positions, length_m=105.0, width_m=68.0)
    assert zs.attacking_third_pct == 100.0
    assert zs.right_pct == 100.0
    assert zs.wide_pct > zs.central_pct


def test_badminton_heatmap_filters_cross_net_positions():
    """Locked near player must not accumulate heat on the far court half."""
    from backend.app.pipeline.pitch_homography import build_calibration

    w, h = 1280, 720
    cal = build_calibration(
        "badminton",
        [[w * 0.2, h * 0.2], [w * 0.8, h * 0.2], [w * 0.8, h * 0.8], [w * 0.2, h * 0.8]],
        image_size=(w, h),
        length_m=13.4,
        width_m=6.1,
    )
    mid = cal.pitch_length_m / 2.0
    near_rows = []
    far_rows = []
    for i in range(20):
        # Homography fans near-court image X across both length halves.
        foot_x = 0.32 if i % 2 == 0 else 0.68
        near_rows.append(
            Row(
                t=float(i),
                frame=i,
                track=1,
                player=None,
                x=foot_x,
                y=0.75,
                bbox=[w * foot_x - 30, 520.0 + i, w * foot_x + 30, 620.0 + i],
                foot_x=foot_x,
                foot_y=0.78,
            )
        )
        far_rows.append(
            Row(
                t=float(i),
                frame=100 + i,
                track=1,
                player=None,
                x=0.5,
                y=0.25,
                bbox=[500.0, 120.0 + i, 560.0, 220.0 + i],
                foot_x=0.42,
                foot_y=0.22,
            )
        )
    mixed = near_rows + far_rows
    unfiltered = build_heatmap_from_rows(mixed, cal, fps=30.0)
    filtered = build_heatmap_from_rows(mixed, cal, fps=30.0, court_side="near")
    assert filtered.sample_count <= unfiltered.sample_count
    assert filtered.sample_count > 0
    grid = bin_positions(
        meter_positions_from_rows(mixed, cal, court_side="near"),
        length_m=cal.pitch_length_m,
        width_m=cal.pitch_width_m,
        grid_cols=53,
        grid_rows=34,
    )
    cell_x = cal.pitch_length_m / 53
    for col in range(53):
        cell_center_x = (col + 0.5) * cell_x
        if cell_center_x > mid + 0.1 and grid[:, col].sum() > 0:
            raise AssertionError("cross-net heat remains after court-side filter")


def test_build_zone_summary_hottest_cell():
    positions = [(10.0, 10.0), (10.2, 10.1), (10.1, 9.9)]
    grid = bin_positions(
        positions,
        length_m=105.0,
        width_m=68.0,
        grid_cols=53,
        grid_rows=34,
    )
    zs = build_zone_summary(
        positions,
        grid,
        length_m=105.0,
        width_m=68.0,
        grid_cols=53,
        grid_rows=34,
    )
    assert zs.hottest_x_m is not None
    assert zs.hottest_y_m is not None
    assert zs.hottest_x_m < 35.0
    assert zs.hottest_y_m < 23.0
