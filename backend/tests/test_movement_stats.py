"""Tests for movement statistics."""

from __future__ import annotations

from backend.app.pipeline.movement_stats import (
    MovementStatsConfig,
    compute_movement_stats,
    compute_movement_stats_from_rows,
)
from backend.app.pipeline.pitch_homography import build_calibration
from backend.app.schemas import Row


def test_constant_speed_metrics():
    points = [(float(i), 0.0, float(i) * 5.0) for i in range(6)]
    stats = compute_movement_stats(points)
    assert stats.distance_m == 25.0
    assert stats.max_speed_m_s == 5.0
    assert stats.avg_speed_m_s == 5.0


def test_sprint_with_row_models():
    rows = [
        Row(t=0.0, frame=0, track=1, player=10, x=0.5, y=0.5, bbox=[0, 0, 10, 10]),
        Row(t=1.0, frame=30, track=1, player=10, x=0.5, y=0.5, bbox=[0, 0, 10, 18]),
        Row(t=2.0, frame=60, track=1, player=10, x=0.5, y=0.5, bbox=[0, 0, 10, 26]),
    ]
    cal = build_calibration(
        "test",
        [[115, 425], [905, 495], [1265, 705], [35, 705]],
        image_size=(1280, 720),
    )
    stats = compute_movement_stats_from_rows(
        rows,
        fps=30.0,
        calibration=cal,
        config=MovementStatsConfig(sprint_speed_m_s=0.5, min_sprint_duration_s=0.5),
    )
    assert stats.units == "meters"
    assert stats.sample_count == 3
    assert stats.distance_m > 0
