"""Sport-specific pipeline gating helpers."""

from backend.app.pipeline.run import _ball_events_enabled
from backend.app.schemas import PlayerDetails


def test_badminton_details_validate():
    details = PlayerDetails.model_validate(
        {
            "sport": "badminton",
            "courtSide": "near",
            "primaryJerseyColor": "#2563eb",
        }
    )
    assert details.sport == "badminton"
    assert details.jerseyNumber == 0


def test_football_ball_events_gated_off_for_badminton(monkeypatch):
    monkeypatch.setenv("ENABLE_BALL_EVENTS", "1")
    is_football = False
    enable_ball = _ball_events_enabled() and is_football
    assert enable_ball is False


def test_football_ball_events_enabled_when_football(monkeypatch):
    monkeypatch.setenv("ENABLE_BALL_EVENTS", "1")
    is_football = True
    enable_ball = _ball_events_enabled() and is_football
    assert enable_ball is True
