"""Tests for movement statistics from trajectories."""

from __future__ import annotations

import math

from movement_stats import (
    MovementStatsConfig,
    compute_movement_stats,
    compute_movement_stats_from_rows,
    foot_position_pixels,
    trajectory_points_from_rows,
)


def test_foot_position_bottom_center():
    assert foot_position_pixels([100.0, 50.0, 200.0, 400.0]) == (150.0, 400.0)


def test_constant_jog_no_sprints():
    # 5 m/s for 5 seconds → below sprint threshold (7 m/s)
    points = [(float(i), 0.0, float(i) * 5.0) for i in range(6)]  # dt=1, dy=5m
    stats = compute_movement_stats(points)
    assert stats.distance_m == 25.0
    assert stats.max_speed_m_s == 5.0
    assert stats.avg_speed_m_s == 5.0
    assert stats.sprint_count == 0
    assert stats.tracked_duration_s == 5.0


def test_sustained_sprint_counted_once():
    # 8 m/s for 2 seconds
    points = [(0.0, 0.0, 0.0), (1.0, 0.0, 8.0), (2.0, 0.0, 16.0)]
    stats = compute_movement_stats(points, config=MovementStatsConfig(sprint_speed_m_s=7.0))
    assert stats.distance_m == 16.0
    assert stats.max_speed_m_s == 8.0
    assert stats.sprint_count == 1


def test_two_sprint_episodes():
    points = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 8.0),
        (2.0, 0.0, 16.0),
        (3.0, 0.0, 16.0),  # pause (0 speed segment)
        (4.0, 0.0, 24.0),
        (5.0, 0.0, 32.0),
    ]
    stats = compute_movement_stats(points, config=MovementStatsConfig(sprint_speed_m_s=7.0))
    assert stats.sprint_count == 2


def test_large_gap_skipped():
    points = [(0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (10.0, 0.0, 2.0)]
    stats = compute_movement_stats(points, config=MovementStatsConfig(max_segment_gap_s=2.0))
    assert stats.distance_m == 1.0
    assert stats.sample_count == 3


def test_avg_speed_uses_active_time_not_wall_clock():
    # 1 m/s for 1 s, 3 s gap (skipped), 1 m/s for 1 s
    points = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (4.0, 2.0, 0.0), (5.0, 3.0, 0.0)]
    stats = compute_movement_stats(points, config=MovementStatsConfig(max_segment_gap_s=2.0))
    assert stats.distance_m == 2.0
    assert stats.avg_speed_m_s == 1.0
    assert stats.tracked_duration_s == 5.0


def test_pixel_units_disable_sprint_count():
    points = [(0.0, 0.0, 0.0), (1.0, 0.0, 800.0), (2.0, 0.0, 1600.0)]
    stats = compute_movement_stats(points, units="pixels")
    assert stats.units == "pixels"
    assert stats.sprint_count == 0


def test_rows_without_calibration_uses_pixels():
    rows = [
        {"frame": 0, "t": 0.0, "bbox": [0.0, 0.0, 10.0, 10.0]},
        {"frame": 1, "t": 1.0, "bbox": [0.0, 0.0, 10.0, 20.0]},
    ]
    stats = compute_movement_stats_from_rows(rows, fps=30.0)
    assert stats.units == "pixels"
    assert stats.distance_m == 10.0


def test_trajectory_sorted_by_frame():
    rows = [
        {"frame": 2, "t": 2.0, "bbox": [0.0, 0.0, 10.0, 20.0]},
        {"frame": 0, "t": 0.0, "bbox": [0.0, 0.0, 10.0, 10.0]},
        {"frame": 1, "t": 1.0, "bbox": [0.0, 0.0, 10.0, 15.0]},
    ]
    points = trajectory_points_from_rows(rows)
    assert [p[0] for p in points] == [0.0, 1.0, 2.0]
    assert points[1][2] == 15.0


def test_empty_rows():
    stats = compute_movement_stats_from_rows([])
    assert stats.distance_m == 0.0
    assert stats.sprint_count == 0
