"""Badminton metrics derived from movement and (future) rally detection."""

from __future__ import annotations

from backend.app.schemas import BadmintonStats, MovementStats, TargetMatch


def build_badminton_stats(
    *,
    movement: MovementStats | None,
    target: TargetMatch,
    has_calibration: bool,
) -> tuple[BadmintonStats | None, str | None]:
    """Return (stats, unavailable_reason). Rally fields stay null until rally detection lands."""
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

    stats = BadmintonStats(
        rally_wins=None,
        total_rallies=None,
        win_rate_pct=None,
        court_coverage_km=court_coverage_km,
        avg_rally_duration_s=None,
        winners=None,
        movement_speed_ms=movement_speed_ms,
    )
    return stats, "rally_detection_pending"
