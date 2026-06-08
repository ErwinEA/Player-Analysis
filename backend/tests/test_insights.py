"""Tests for LLM insights facts packet and orchestrator."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.pipeline.insights import (
    InsightsUnavailableError,
    build_facts_packet,
    format_insights_text,
    generate_insights,
    trim_inferred_events,
)
from backend.app.schemas import (
    HeatmapZoneSummary,
    InferredBallEvent,
    InferredEventsSummary,
    InsightsPlayerContext,
    InsightsRequest,
    MovementStats,
    PlayerEventCounts,
    TargetMatch,
)


def _sample_request(**overrides) -> InsightsRequest:
    base = InsightsRequest(
        player=InsightsPlayerContext(name="Alex", jerseyNumber=10, teamName="Home"),
        target=TargetMatch(jersey=10, track_id=1, confidence=0.9, method="number"),
        movement=MovementStats(
            distance_m=1200.0,
            max_speed_m_s=7.5,
            avg_speed_m_s=2.1,
            sprint_count=3,
            tracked_duration_s=90.0,
            sample_count=200,
        ),
        event_counts=PlayerEventCounts(Pass=5, Shot=2, Goal=0, Drive=1),
        inferred_events_summary=InferredEventsSummary(
            weak_count=1,
            total_count=2,
            events=[
                InferredBallEvent(frame=100, kind="Pass", lock_confidence="strong"),
                InferredBallEvent(frame=200, kind="Shot", lock_confidence="weak"),
            ],
        ),
        zone_summary=HeatmapZoneSummary(
            defensive_third_pct=10.0,
            middle_third_pct=40.0,
            attacking_third_pct=50.0,
            left_pct=20.0,
            center_pct=50.0,
            right_pct=30.0,
            central_pct=60.0,
            wide_pct=40.0,
            hottest_x_m=78.0,
            hottest_y_m=12.0,
        ),
        heatmap_source="locked_target",
        provenance="inferred",
        drive_contact_m=15.0,
        warnings=["Player lock was uncertain."],
    )
    return base.model_copy(update=overrides)


def test_trim_inferred_events_caps_at_10():
    events = [
        InferredBallEvent(frame=i, kind="Pass", lock_confidence="strong")
        for i in range(15)
    ]
    total, weak, trimmed = trim_inferred_events(events, max_events=10)
    assert total == 15
    assert weak == 0
    assert len(trimmed) == 10


def test_build_facts_packet_includes_zone_summary():
    facts = build_facts_packet(_sample_request())
    assert "defensive 10.0%" in facts
    assert "Hottest zone: 78.0m x 12.0m" in facts
    assert "Passes: 5" in facts
    assert "Alex" in facts


def test_build_facts_packet_missing_events():
    facts = build_facts_packet(
        _sample_request(
            event_counts=None,
            inferred_events_summary=None,
            provenance=None,
        )
    )
    assert "Events: not available" in facts
    assert "defensive 10.0%" in facts


def test_build_facts_packet_missing_zone():
    facts = build_facts_packet(_sample_request(zone_summary=None))
    assert "Positioning (heatmap zones): not available" in facts


def test_format_insights_text_splits_crowded_bullets():
    raw = (
        "Here's a brief gameplay analysis based solely on the provided data: "
        "* First point here. * Second point here. "
        "Caveat: Events are inferred only."
    )
    formatted = format_insights_text(raw)
    assert formatted.startswith("• First point here.")
    assert "\n\n• Second point here." in formatted
    assert formatted.endswith("Caveat: Events are inferred only.")
    assert "Here's a brief" not in formatted


@patch("backend.app.pipeline.insights.call_ollama_chat")
def test_generate_insights_success(mock_chat):
    mock_chat.return_value = "- Strong attacking presence\nCaveat: inferred events only."
    result = generate_insights(_sample_request())
    assert result.summary is not None
    assert "attacking" in result.summary
    assert result.unavailable_reason is None
    mock_chat.assert_called_once()


@patch("backend.app.pipeline.insights.call_ollama_chat")
def test_generate_insights_ollama_unavailable_returns_reason(mock_chat):
    mock_chat.side_effect = InsightsUnavailableError("Ollama is not reachable.")
    result = generate_insights(_sample_request())
    assert result.summary is None
    assert result.unavailable_reason is not None
    assert "reachable" in result.unavailable_reason.lower()


@patch("backend.app.pipeline.insights.call_ollama_chat")
def test_generate_insights_empty_output(mock_chat):
    mock_chat.return_value = "   "
    result = generate_insights(_sample_request())
    assert result.summary is None
    assert result.unavailable_reason is not None


@patch("backend.app.pipeline.insights.call_ollama_chat")
def test_insights_endpoint_returns_200_when_ollama_down(mock_chat):
    mock_chat.side_effect = InsightsUnavailableError("Ollama is not reachable.")
    client = TestClient(app)
    res = client.post("/api/insights", json=_sample_request().model_dump())
    assert res.status_code == 200
    body = res.json()
    assert body["summary"] is None
    assert "reachable" in body["unavailable_reason"].lower()


def test_health_includes_ollama():
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert "ollama" in body
    assert "reachable" in body["ollama"]
    assert "model" in body["ollama"]
