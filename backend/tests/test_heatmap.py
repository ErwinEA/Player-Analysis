"""Tests for player position heatmap generation."""

from __future__ import annotations

import numpy as np

from backend.app.pipeline.heatmap import (
    bin_positions,
    build_heatmap_from_rows,
    meter_positions_from_rows,
    pitch_grid_shape,
    smooth_grid,
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
