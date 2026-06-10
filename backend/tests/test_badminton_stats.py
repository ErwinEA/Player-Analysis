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
    assert reason == "no_shuttle_weights"


def test_build_badminton_stats_pending_when_detector_ran_without_rallies():
    target = TargetMatch(jersey=0, track_id=7, confidence=0.8, method="court_side")
    stats, reason = build_badminton_stats(
        movement=None,
        target=target,
        has_calibration=True,
        rally_events=[],
        shuttle_available=True,
    )
    assert stats is not None
    assert stats.total_rallies is None
    assert reason == "rally_detection_pending"


def test_build_badminton_stats_fills_rally_fields():
    from backend.app.pipeline.badminton.rally_tracker import RallyEvent

    target = TargetMatch(jersey=0, track_id=7, confidence=1.0, method="court_side")
    events = [
        RallyEvent(
            start_frame=0,
            end_frame=300,
            winner_side="near",
            locked_player_touched_last=True,
        ),
        RallyEvent(
            start_frame=400,
            end_frame=550,
            winner_side="far",
            locked_player_touched_last=False,
        ),
        RallyEvent(
            start_frame=600,
            end_frame=900,
            winner_side="near",
            locked_player_touched_last=False,
        ),
    ]
    stats, reason = build_badminton_stats(
        movement=None,
        target=target,
        has_calibration=True,
        rally_events=events,
        court_side="near",
        fps=30.0,
        shuttle_available=True,
    )
    assert reason is None
    assert stats is not None
    assert stats.total_rallies == 3
    assert stats.rally_wins == 2
    assert stats.win_rate_pct == 66.7
    assert stats.avg_rally_duration_s == 8.33
    assert stats.winners == 1
