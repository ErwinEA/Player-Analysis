"""LLM gameplay insights via local Ollama."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import httpx

from backend.app.schemas import (
    BadmintonStats,
    HeatmapZoneSummary,
    InferredBallEvent,
    InsightsRequest,
    InsightsResponse,
    MovementStats,
)

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_INSIGHTS_MODEL = "gemma3:1b"
DEFAULT_INSIGHTS_TIMEOUT_S = 60.0
OLLAMA_HEALTH_TIMEOUT_S = 2.0
MAX_INFERRED_EVENTS = 10


class InsightsUnavailableError(Exception):
    """Ollama unreachable, timed out, or returned unusable output."""


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name, "").strip()
    return raw or default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def insights_model() -> str:
    return _env_str("INSIGHTS_MODEL", DEFAULT_INSIGHTS_MODEL)


def ollama_base_url() -> str:
    return _env_str("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")


def insights_timeout_s() -> float:
    return _env_float("INSIGHTS_TIMEOUT_S", DEFAULT_INSIGHTS_TIMEOUT_S)


def trim_inferred_events(
    events: list[InferredBallEvent] | None,
    *,
    max_events: int = MAX_INFERRED_EVENTS,
) -> tuple[int, int, list[InferredBallEvent]]:
    """Return (total_count, weak_count, capped events list)."""
    if not events:
        return 0, 0, []
    weak_count = sum(1 for e in events if e.lock_confidence == "weak")
    return len(events), weak_count, events[:max_events]


def _format_movement(movement: MovementStats | None) -> list[str]:
    if movement is None or movement.units != "meters":
        return ["Movement: not available"]
    return [
        "Movement:",
        f"- Distance: {movement.distance_m:.1f} m ({movement.distance_m / 1000:.2f} km)",
        f"- Avg speed: {movement.avg_speed_m_s:.2f} m/s",
        f"- Max speed: {movement.max_speed_m_s:.2f} m/s",
        f"- Sprints: {movement.sprint_count}",
        f"- Tracked duration: {movement.tracked_duration_s:.1f} s",
        f"- Samples: {movement.sample_count}",
    ]


def _format_zone_summary(
    zone: HeatmapZoneSummary | None,
    *,
    sport: str = "football",
) -> list[str]:
    if zone is None:
        return ["Positioning (heatmap zones): not available"]
    length_label = "Court length" if sport == "badminton" else "Pitch"
    coord_label = "court coords" if sport == "badminton" else "pitch coords"
    lines = [
        "Positioning (heatmap zones):",
        (
            f"- {length_label} thirds: defensive {zone.defensive_third_pct}%, "
            f"middle {zone.middle_third_pct}%, attacking {zone.attacking_third_pct}%"
        ),
        (
            f"- Width channels: left {zone.left_pct}%, "
            f"center {zone.center_pct}%, right {zone.right_pct}%"
        ),
        f"- Central corridor: {zone.central_pct}%, wide areas: {zone.wide_pct}%",
    ]
    if zone.hottest_x_m is not None and zone.hottest_y_m is not None:
        lines.append(
            f"- Hottest zone: {zone.hottest_x_m}m x {zone.hottest_y_m}m ({coord_label})"
        )
    return lines


def _format_badminton_stats(stats: BadmintonStats | None) -> list[str]:
    if stats is None or stats.total_rallies is None:
        return ["Rally stats: not available"]
    lines = [
        "Rally stats (heuristic — not official scoring):",
        f"- Total rallies (estimated): {stats.total_rallies}",
    ]
    if stats.avg_rally_duration_s is not None:
        lines.append(
            f"- Avg rally duration (estimated): {stats.avg_rally_duration_s:.1f} s"
        )
    if stats.longest_rally_duration_s is not None:
        lines.append(
            f"- Longest rally (estimated): {stats.longest_rally_duration_s:.1f} s"
        )
    return lines


def build_facts_packet(req: InsightsRequest) -> str:
    """Compact facts block for the LLM (~300-600 tokens)."""
    lines: list[str] = []
    sport = req.sport

    player = req.player
    if sport == "badminton":
        name = player.name.strip() or "unnamed player"
        team = player.teamName.strip() or "unknown club/country"
        lines.append("Player:")
        lines.append(f"- Name: {name}")
        lines.append(f"- Club/country: {team}")
        if req.court_side:
            lines.append(f"- Court side: {req.court_side}")
    else:
        name = player.name.strip() or f"#{player.jerseyNumber}"
        team = player.teamName.strip() or "unknown team"
        lines.append("Player:")
        lines.append(f"- Name: {name}")
        lines.append(f"- Jersey: {player.jerseyNumber}")
        lines.append(f"- Team: {team}")

    target = req.target
    lines.append("Player lock:")
    lines.append(f"- Track ID: {target.track_id}")
    lines.append(f"- Method: {target.method or 'none'}")
    lines.append(f"- Confidence: {target.confidence:.2f}")
    if req.heatmap_source:
        lines.append(f"- Heatmap source: {req.heatmap_source}")

    lines.extend(_format_movement(req.movement))

    if sport == "badminton":
        lines.extend(_format_badminton_stats(req.badminton_stats))
    elif req.event_counts is None or req.provenance != "inferred":
        lines.append("Events: not available")
    else:
        ec = req.event_counts
        lines.append("Events (inferred from ball tracking):")
        lines.append(f"- Passes: {ec.Pass}, Shots: {ec.Shot}, Goals: {ec.Goal}, Drives: {ec.Drive}")
        if req.drive_contact_m is not None and req.drive_contact_m > 0:
            lines.append(f"- Drive ball contact: {req.drive_contact_m:.0f} m")
        summary = req.inferred_events_summary
        if summary is not None:
            lines.append(
                f"- Event timeline: {summary.total_count} events "
                f"({summary.weak_count} weak/uncertain)"
            )
            for ev in summary.events:
                phase = f", phase={ev.detection_phase}" if ev.detection_phase else ""
                lines.append(
                    f"  - frame {ev.frame}: {ev.kind} "
                    f"({ev.lock_confidence}{phase})"
                )
            if summary.total_count > len(summary.events):
                omitted = summary.total_count - len(summary.events)
                lines.append(f"  - ... {omitted} more events omitted")

    lines.extend(_format_zone_summary(req.zone_summary, sport=sport))

    if req.warnings:
        lines.append("Warnings:")
        for w in req.warnings:
            lines.append(f"- {w}")

    return "\n".join(lines)


def format_insights_text(raw: str) -> str:
    """Normalize LLM output so each bullet and caveat is on its own line."""
    text = raw.strip()
    text = re.sub(
        r"^Here(?:'s| is) a brief gameplay analysis[^:\n]*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*[*•-]\s+", "\n\n• ", text)
    text = re.sub(r"\s*Caveat:\s*", "\n\nCaveat: ", text, flags=re.IGNORECASE)
    blocks = [re.sub(r"\s+", " ", block.strip()) for block in text.split("\n\n") if block.strip()]
    return "\n\n".join(blocks)


def build_prompt(facts: str, *, sport: str = "football") -> list[dict[str, str]]:
    if sport == "badminton":
        system = (
            "You are a badminton performance analyst. Use ONLY the facts provided. "
            "Do not invent statistics, rallies, or points. Rally counts and durations "
            "are heuristic estimates only — not official scoring. There is no line-call "
            "or serve/return detection; do not describe in/out calls or serve splits. "
            "Rally counts may include replays in highlight clips. If rally stats are "
            "missing or warnings note uncertainty, state that clearly in a caveat. "
            "Court is 13.4 m long × 6.1 m (singles). Length thirds split the long "
            "axis (0 m and 13.4 m are baselines; net is near 6.7 m) — not necessarily "
            "the player's attacking direction."
        )
        bullets = (
            "positioning/court coverage, movement workload, rally/point patterns "
            "(only if rally stats are present)"
        )
    else:
        system = (
            "You are a football performance analyst. Use ONLY the facts provided. "
            "Do not invent statistics or events. If data is missing or warnings "
            "note uncertainty, state that clearly in a caveat. "
            "Pitch thirds are geometric (0m = one goal line, 105m = other) — "
            "not necessarily the team's attacking direction."
        )
        bullets = "positioning, involvement, movement"
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"{facts}\n\n"
                "Write a brief gameplay analysis with strict formatting:\n"
                f"- 3 to 4 bullet points ({bullets})\n"
                "- 1 final line starting with 'Caveat:'\n"
                "- Each bullet on its own line, starting with '• '\n"
                "- Blank line between each bullet\n"
                "- Blank line before 'Caveat:'\n"
                "- No intro sentence. 1-2 sentences per bullet. Plain text only."
            ),
        },
    ]


def call_ollama_chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    base_url: str | None = None,
    timeout_s: float | None = None,
) -> str:
    model = model or insights_model()
    base_url = (base_url or ollama_base_url()).rstrip("/")
    timeout_s = timeout_s if timeout_s is not None else insights_timeout_s()
    url = f"{base_url}/api/chat"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.3},
    }
    try:
        with httpx.Client(timeout=timeout_s) as client:
            res = client.post(url, json=payload)
    except httpx.ConnectError as exc:
        raise InsightsUnavailableError(
            "Ollama is not reachable. Start it with `ollama serve`."
        ) from exc
    except httpx.TimeoutException as exc:
        raise InsightsUnavailableError(
            f"Ollama timed out after {timeout_s:.0f}s."
        ) from exc
    except httpx.HTTPError as exc:
        raise InsightsUnavailableError(f"Ollama request failed: {exc}") from exc

    if res.status_code != 200:
        detail = res.text[:200] if res.text else res.status_code
        raise InsightsUnavailableError(f"Ollama returned HTTP {res.status_code}: {detail}")

    try:
        body = res.json()
    except ValueError as exc:
        raise InsightsUnavailableError("Ollama returned invalid JSON.") from exc

    message = body.get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise InsightsUnavailableError("Ollama returned empty content.")
    return content.strip()


def ollama_reachable(
    *,
    base_url: str | None = None,
    timeout_s: float = OLLAMA_HEALTH_TIMEOUT_S,
) -> bool:
    base_url = (base_url or ollama_base_url()).rstrip("/")
    try:
        with httpx.Client(timeout=timeout_s) as client:
            res = client.get(f"{base_url}/api/tags")
        return res.status_code == 200
    except httpx.HTTPError:
        return False


def ollama_health() -> dict[str, object]:
    reachable = ollama_reachable()
    return {
        "reachable": reachable,
        "model": insights_model(),
        "base_url": ollama_base_url(),
    }


def generate_insights(req: InsightsRequest) -> InsightsResponse:
    facts = build_facts_packet(req)
    try:
        messages = build_prompt(facts, sport=req.sport)
        summary = call_ollama_chat(messages).strip()
    except InsightsUnavailableError as exc:
        logger.warning("Insights unavailable: %s", exc)
        return InsightsResponse(summary=None, unavailable_reason=str(exc))
    if not summary:
        return InsightsResponse(
            summary=None,
            unavailable_reason="Ollama returned empty insights.",
        )
    return InsightsResponse(
        summary=format_insights_text(summary),
        unavailable_reason=None,
    )
