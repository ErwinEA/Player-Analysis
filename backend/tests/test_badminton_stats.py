"""Badminton player details and stats helpers."""

from backend.app.pipeline.badminton_stats import build_badminton_stats
from backend.app.schemas import MovementStats, PlayerDetails, TargetMatch


def test_badminton_player_details_requires_court_side_and_color():
    details = PlayerDetails.model_validate(
        {
            "sport": "badminton",
            "courtSide": "near",
            "primaryJerseyColor": "#2563eb",
        }
    )
    assert details.sport == "badminton"
    assert details.jerseyNumber == 0


def test_football_player_details_requires_jersey():
    try:
        PlayerDetails.model_validate({"sport": "football", "jerseyNumber": 0})
    except ValueError as exc:
        assert "jerseyNumber" in str(exc)
    else:
        raise AssertionError("expected validation error")


def test_build_badminton_stats_from_movement():
    movement = MovementStats(
        distance_m=1250.0,
        max_speed_m_s=4.2,
        avg_speed_m_s=2.5,
        sprint_count=3,
        tracked_duration_s=500.0,
        sample_count=400,
        units="meters",
    )
    target = TargetMatch(jersey=0, track_id=7, confidence=0.8, method="color")
    stats, reason = build_badminton_stats(
        movement=movement,
        target=target,
        has_calibration=True,
    )
    assert stats is not None
    assert stats.court_coverage_km == 1.25
    assert stats.movement_speed_ms == 2.5
    assert stats.rally_wins is None
    assert reason == "rally_detection_pending"
