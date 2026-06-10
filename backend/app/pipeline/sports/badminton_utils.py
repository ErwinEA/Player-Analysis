"""Badminton player disambiguation and court geometry helpers."""

from __future__ import annotations

import os
from typing import Any, Literal

from backend.app.pipeline.pitch_homography import PitchCalibration

CourtSide = Literal["near", "far"]
RallyState = Literal["IDLE", "LIVE", "END"]

_COLOR_LOCK_THRESHOLD = float(os.environ.get("COLOR_LOCK_THRESHOLD", "0.15"))
_COLOR_MIN_FLOOR = float(os.environ.get("COLOR_MATCH_MIN", "0.05"))
_COLOR_MIN_SAMPLES = 5


def centroid_y(bbox: list[float]) -> float:
    return (float(bbox[1]) + float(bbox[3])) / 2.0


def net_line_y_px(calibration: PitchCalibration) -> float:
    """Pixel Y of the court centre line (net) from homography."""
    length_m = calibration.pitch_length_m
    width_m = calibration.pitch_width_m
    px, py = calibration.meters_to_pixel(length_m / 2.0, width_m / 2.0)
    return float(py)


def _on_near_side(cy: float, net_y_px: float) -> bool:
    """Near court = closer to camera = larger image Y (below net line)."""
    return cy > net_y_px


def disambiguate_by_court_side(
    tracks: list[dict[str, Any]],
    court_side: CourtSide,
    net_y_px: float,
    *,
    min_tracks: int = 2,
) -> int | None:
    """Return track_id on the selected court side when both players are visible.

    MVP: requires exactly one mature track on the chosen side (singles, both visible).
    """
    if len(tracks) < min_tracks or not court_side:
        return None

    candidates: list[tuple[int, float]] = []
    for tr in tracks:
        bbox = tr.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        cy = centroid_y(bbox)
        on_near = _on_near_side(cy, net_y_px)
        if court_side == "near" and on_near:
            candidates.append((int(tr["track_id"]), cy))
        elif court_side == "far" and not on_near:
            candidates.append((int(tr["track_id"]), cy))

    if len(candidates) != 1:
        return None
    return candidates[0][0]


def best_color_track_id(color_scores: dict[int, list[float]]) -> int | None:
    """Track with highest average shirt-color score (matcher color-lock thresholds)."""
    best_id: int | None = None
    best_avg = 0.0
    min_avg = max(_COLOR_LOCK_THRESHOLD, _COLOR_MIN_FLOOR)
    for track_id, scores in color_scores.items():
        if len(scores) < _COLOR_MIN_SAMPLES:
            continue
        avg = sum(scores) / len(scores)
        if avg > best_avg:
            best_id = track_id
            best_avg = avg
    if best_id is None or best_avg < min_avg:
        return None
    return best_id


def disambiguate_with_fallback(
    tracks: list[dict[str, Any]],
    court_side: CourtSide,
    net_y_px: float,
    color_scores: dict[int, list[float]],
    *,
    min_tracks: int = 2,
) -> tuple[int | None, str]:
    """Court-side lock with shirt-color fallback. Returns (track_id, method_hint)."""
    track_id = disambiguate_by_court_side(
        tracks, court_side, net_y_px, min_tracks=min_tracks
    )
    if track_id is not None:
        return track_id, "court_side"

    color_id = best_color_track_id(color_scores)
    if color_id is not None:
        return color_id, "color"
    return None, "none"
