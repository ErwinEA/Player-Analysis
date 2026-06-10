"""Badminton metrics derived from movement and rally detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.app.schemas import BadmintonStats, MovementStats, TargetMatch

if TYPE_CHECKING:
    from backend.app.pipeline.badminton.rally_tracker import RallyEvent


def build_badminton_stats(
    *,
    movement: MovementStats | None,
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

    court_coverage_km: float | None = None
    movement_speed_ms: float | None = None
    if movement is not None and movement.units == "meters":
        court_coverage_km = round(movement.distance_m / 1000.0, 3)
        movement_speed_ms = round(movement.avg_speed_m_s, 3)

    rally_wins: int | None = None
    total_rallies: int | None = None
    win_rate_pct: float | None = None
    avg_rally_duration_s: float | None = None
    winners: int | None = None
    reason: str | None

    if rally_events:
        total_rallies = len(rally_events)
        rally_wins = sum(
            1 for ev in rally_events if ev.winner_side == court_side
        )
        win_rate_pct = round(rally_wins / total_rallies * 100.0, 1)
        if fps > 0:
            avg_rally_duration_s = round(
                sum((ev.end_frame - ev.start_frame) / fps for ev in rally_events)
                / total_rallies,
                2,
            )
        winners = sum(
            1
            for ev in rally_events
            if ev.winner_side == court_side and ev.locked_player_touched_last
        )
        reason = None
    elif not shuttle_available:
        reason = "no_shuttle_weights"
    else:
        # Detector ran but found no rallies in this clip.
        reason = "rally_detection_pending"

    stats = BadmintonStats(
        rally_wins=rally_wins,
        total_rallies=total_rallies,
        win_rate_pct=win_rate_pct,
        court_coverage_km=court_coverage_km,
        avg_rally_duration_s=avg_rally_duration_s,
        winners=winners,
        movement_speed_ms=movement_speed_ms,
    )
    return stats, reason
