from backend.app.pipeline.movement_stats import (
    compute_movement_stats_from_metres,
    trajectory_points_from_metres,
)


def test_trajectory_points_from_metres():
    pts = trajectory_points_from_metres([(0.0, 10.0, 20.0), (1.0, 15.0, 25.0)])
    assert len(pts) == 2
    assert pts[0][1] == 10.0


def test_compute_movement_stats_from_metres():
    stats = compute_movement_stats_from_metres(
        [(0.0, 0.0, 0.0), (1.0, 10.0, 0.0), (2.0, 20.0, 0.0)]
    )
    assert stats.distance_m == 20.0
    assert stats.units == "meters"
