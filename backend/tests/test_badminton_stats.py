"""Badminton player details and stats helpers."""

from backend.app.pipeline.badminton_stats import build_badminton_stats
from backend.app.schemas import PlayerDetails, TargetMatch


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


def test_build_badminton_stats_requires_shuttle_when_no_rallies():
    target = TargetMatch(jersey=0, track_id=7, confidence=0.8, method="color")
    stats, reason = build_badminton_stats(
        target=target,
        has_calibration=True,
    )
    assert stats is not None
    assert stats.total_rallies is None
    assert reason == "no_shuttle_weights"


def test_build_badminton_stats_pending_when_detector_ran_without_rallies():
    target = TargetMatch(jersey=0, track_id=7, confidence=0.8, method="court_side")
    stats, reason = build_badminton_stats(
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
            serving_side="near",
            locked_player_touched_last=True,
        ),
        RallyEvent(
            start_frame=400,
            end_frame=550,
            winner_side="far",
            serving_side="far",
            locked_player_touched_last=False,
        ),
        RallyEvent(
            start_frame=600,
            end_frame=900,
            winner_side="near",
            serving_side="near",
            locked_player_touched_last=False,
        ),
        RallyEvent(
            start_frame=1000,
            end_frame=1180,
            winner_side="near",
            serving_side="far",
            locked_player_touched_last=False,
        ),
    ]
    stats, reason = build_badminton_stats(
        target=target,
        has_calibration=True,
        rally_events=events,
        court_side="near",
        fps=30.0,
        shuttle_available=True,
    )
    assert reason is None
    assert stats is not None
    assert stats.total_rallies == 4
    assert stats.avg_rally_duration_s == 7.75
    assert stats.longest_rally_duration_s == 10.0
    assert stats.total_points_won == 3
    assert stats.points_won_on_serve == 2
    assert stats.points_won_on_return == 1
    assert stats.points_in == 3
    assert stats.points_out == 1


def test_build_badminton_stats_clamps_duration_outliers(monkeypatch):
    from backend.app.pipeline.badminton.rally_tracker import RallyEvent

    monkeypatch.setenv("BADMINTON_MIN_RALLY_S", "1.0")
    monkeypatch.setenv("BADMINTON_MAX_RALLY_S", "120.0")
    target = TargetMatch(jersey=0, track_id=7, confidence=1.0, method="court_side")
    events = [
        RallyEvent(0, 15, "near", "near", False),  # 0.5s — excluded
        RallyEvent(100, 400, "near", "near", False),  # 10s — included
    ]
    stats, _ = build_badminton_stats(
        target=target,
        has_calibration=True,
        rally_events=events,
        court_side="near",
        fps=30.0,
        shuttle_available=True,
    )
    assert stats is not None
    assert stats.avg_rally_duration_s == 10.0
    assert stats.longest_rally_duration_s == 10.0
