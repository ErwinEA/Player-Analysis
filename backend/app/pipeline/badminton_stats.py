"""Badminton metrics derived from rally detection."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from backend.app.schemas import BadmintonStats, TargetMatch

if TYPE_CHECKING:
    from backend.app.pipeline.badminton.rally_tracker import RallyEvent


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _rally_duration_s(ev: "RallyEvent", fps: float) -> float:
    if fps <= 0:
        return 0.0
    return (ev.end_frame - ev.start_frame) / fps


def _valid_rally_durations(
    rally_events: list["RallyEvent"],
    fps: float,
) -> list[float]:
    min_s = _env_float("BADMINTON_MIN_RALLY_S", 1.0)
    max_s = _env_float("BADMINTON_MAX_RALLY_S", 120.0)
    out: list[float] = []
    for ev in rally_events:
        d = _rally_duration_s(ev, fps)
        if min_s <= d <= max_s:
            out.append(d)
    return out


def build_badminton_stats(
    *,
    target: TargetMatch,
    has_calibration: bool,
    rally_events: list["RallyEvent"] | None = None,
    court_side: str = "",
    fps: float = 30.0,
    shuttle_available: bool = False,
) -> tuple[BadmintonStats | None, str | None]:
    """Return (stats, unavailable_reason). Rally fields fill when rally detection ran."""
    locked = target.track_id is not None and target.method not in (None, "none")

    if not locked:
        return None, "no_lock"
    if not has_calibration:
        return None, "no_calibration"

    total_rallies: int | None = None
    avg_rally_duration_s: float | None = None
    longest_rally_duration_s: float | None = None
    points_won_on_serve: int | None = None
    points_won_on_return: int | None = None
    total_points_won: int | None = None
    points_in: int | None = None
    points_out: int | None = None
    reason: str | None

    if rally_events:
        total_rallies = len(rally_events)
        durations = _valid_rally_durations(rally_events, fps)
        if durations:
            avg_rally_duration_s = round(sum(durations) / len(durations), 2)
            longest_rally_duration_s = round(max(durations), 2)

        serve_wins = 0
        return_wins = 0
        won = 0
        lost = 0
        for ev in rally_events:
            if ev.winner_side == court_side:
                won += 1
                if ev.serving_side == court_side:
                    serve_wins += 1
                else:
                    return_wins += 1
            elif ev.winner_side not in (None, "", "unknown"):
                lost += 1

        points_won_on_serve = serve_wins
        points_won_on_return = return_wins
        total_points_won = won
        points_in = won
        points_out = lost
        reason = None
    elif not shuttle_available:
        reason = "no_shuttle_weights"
    else:
        reason = "rally_detection_pending"

    stats = BadmintonStats(
        total_rallies=total_rallies,
        avg_rally_duration_s=avg_rally_duration_s,
        longest_rally_duration_s=longest_rally_duration_s,
        points_won_on_serve=points_won_on_serve,
        points_won_on_return=points_won_on_return,
        total_points_won=total_points_won,
        points_in=points_in,
        points_out=points_out,
    )
    return stats, reason
