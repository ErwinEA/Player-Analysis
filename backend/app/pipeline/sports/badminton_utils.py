"""Badminton player disambiguation and court geometry helpers."""

from __future__ import annotations

from typing import Any, Literal

from backend.app.pipeline.matcher import color_lock_min_avg, color_lock_min_samples
from backend.app.pipeline.pitch_homography import PitchCalibration, is_badminton_court

CourtSide = Literal["near", "far"]
RallyState = Literal["IDLE", "LIVE", "END"]

_NET_MARGIN_M = 0.08


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
    min_avg = color_lock_min_avg()
    min_samples = color_lock_min_samples()
    for track_id, scores in color_scores.items():
        if len(scores) < min_samples:
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


def net_line_x_m(calibration: PitchCalibration) -> float:
    """Net position along court length in top-down metres (vertical net on template)."""
    return calibration.pitch_length_m / 2.0


CourtHalf = Literal["low", "high"]


def _row_image_y(
    row: Any,
    image_height: int,
) -> float | None:
    """Image Y for court-side checks: foot when available, else bbox centroid."""
    foot_y = row.foot_y if hasattr(row, "foot_y") else row.get("foot_y")
    if foot_y is not None:
        return float(foot_y) * image_height
    bbox = row.bbox if hasattr(row, "bbox") else row.get("bbox")
    if bbox and len(bbox) >= 4:
        return centroid_y(bbox)
    return None


def row_on_court_side(
    row: Any,
    court_side: CourtSide | str,
    net_y_px: float,
    *,
    image_height: int,
) -> bool:
    """True when the row's foot/centroid lies on the selected broadcast court side."""
    if court_side not in ("near", "far"):
        return True
    py = _row_image_y(row, image_height)
    if py is None:
        return False
    on_near = _on_near_side(py, net_y_px)
    return on_near if court_side == "near" else not on_near


def filter_rows_by_court_side(
    rows: list[Any],
    calibration: PitchCalibration,
    court_side: CourtSide | str,
) -> list[Any]:
    """Drop rows whose image foot position is on the opponent's court side."""
    if court_side not in ("near", "far"):
        return rows
    if not is_badminton_court(calibration.pitch_length_m, calibration.pitch_width_m):
        return rows
    net_y = net_line_y_px(calibration)
    _, image_height = calibration.image_size
    return [
        row
        for row in rows
        if row_on_court_side(
            row, court_side, net_y, image_height=image_height
        )
    ]


def median_foot_x_px(
    rows: list[Any],
    calibration: PitchCalibration,
) -> float | None:
    """Median image X of foot/centroid positions for homography half probing."""
    width, height = calibration.image_size
    xs: list[float] = []
    for row in rows:
        foot_x = row.foot_x if hasattr(row, "foot_x") else row.get("foot_x")
        if foot_x is not None:
            xs.append(float(foot_x) * width)
            continue
        bbox = row.bbox if hasattr(row, "bbox") else row.get("bbox")
        if bbox and len(bbox) >= 4:
            xs.append((float(bbox[0]) + float(bbox[2])) / 2.0)
    if not xs:
        return None
    # Mean resists homography fan-out that alternates left/right image X.
    return sum(xs) / len(xs)


def court_side_to_metre_half(
    calibration: PitchCalibration,
    court_side: CourtSide | str,
    *,
    image_probe_x: float | None = None,
) -> CourtHalf | None:
    """Map broadcast near/far image side to low/high metre X via homography probe."""
    if court_side not in ("near", "far"):
        return None
    if not is_badminton_court(calibration.pitch_length_m, calibration.pitch_width_m):
        return None
    width, height = calibration.image_size
    net_y = net_line_y_px(calibration)
    offset = height * 0.06
    py = net_y + offset if court_side == "near" else net_y - offset
    px = image_probe_x if image_probe_x is not None else width / 2.0
    try:
        x_m, _ = calibration.pixel_to_meters(px, py)
    except ValueError:
        return None
    return "low" if x_m < net_line_x_m(calibration) else "high"


def dominant_court_half(
    positions: list[tuple[float, float]],
    calibration: PitchCalibration,
) -> CourtHalf:
    """Which side of the net (low/high X) holds most metre samples for one player."""
    mid_x = net_line_x_m(calibration)
    low = sum(1 for x_m, _ in positions if x_m < mid_x)
    return "low" if low * 2 >= len(positions) else "high"


def metre_half_for_court_side(
    positions: list[tuple[float, float]],
    calibration: PitchCalibration,
    court_side: CourtSide | str,
    *,
    image_probe_x: float | None = None,
) -> CourtHalf:
    """Pick metre court half from player samples, steered by court_side when sparse."""
    mid_x = net_line_x_m(calibration)
    margin = _NET_MARGIN_M
    if len(positions) >= 4:
        low_count = sum(1 for x, _ in positions if x < mid_x - margin)
        high_count = sum(1 for x, _ in positions if x > mid_x + margin)
        minority = min(low_count, high_count)
        if minority > 0 and minority / len(positions) >= 0.3:
            mapped = court_side_to_metre_half(
                calibration, court_side, image_probe_x=image_probe_x
            )
            if mapped is not None:
                return mapped
    if len(positions) >= 3:
        xs = sorted(x for x, _ in positions)
        median_x = xs[len(xs) // 2]
        return "low" if median_x < mid_x else "high"
    mapped = court_side_to_metre_half(
        calibration, court_side, image_probe_x=image_probe_x
    )
    if mapped is not None:
        return mapped
    if len(positions) >= 2:
        return dominant_court_half(positions, calibration)
    return "low"


def position_in_court_half(
    x_m: float,
    calibration: PitchCalibration,
    half: CourtHalf,
) -> bool:
    """True when metre X lies on the chosen court half (net at length/2)."""
    mid_x = net_line_x_m(calibration)
    margin = _NET_MARGIN_M
    if half == "low":
        return x_m < mid_x - margin
    return x_m > mid_x + margin


def filter_positions_to_court_half(
    positions: list[tuple[float, float]],
    calibration: PitchCalibration,
    court_side: CourtSide | str,
    *,
    image_probe_x: float | None = None,
) -> list[tuple[float, float]]:
    """Keep metre samples on the court half for the selected near/far side."""
    if court_side not in ("near", "far"):
        return positions
    if not is_badminton_court(calibration.pitch_length_m, calibration.pitch_width_m):
        return positions
    if not positions:
        return positions
    half = metre_half_for_court_side(
        positions, calibration, court_side, image_probe_x=image_probe_x
    )
    filtered = [
        (x_m, y_m)
        for x_m, y_m in positions
        if position_in_court_half(x_m, calibration, half)
    ]
    # If calibration is poor, homography half may reject almost everything — keep majority.
    if len(positions) >= 4 and len(filtered) < len(positions) * 0.15:
        half = dominant_court_half(positions, calibration)
        filtered = [
            (x_m, y_m)
            for x_m, y_m in positions
            if position_in_court_half(x_m, calibration, half)
        ]
    return filtered


def mask_grid_to_court_half(
    grid: Any,
    calibration: PitchCalibration,
    half: CourtHalf,
) -> Any:
    """Zero grid cells on the opponent's court half (prevents Gaussian net bleed)."""
    if not is_badminton_court(calibration.pitch_length_m, calibration.pitch_width_m):
        return grid
    _, cols = grid.shape
    length_m = calibration.pitch_length_m
    mid_x = net_line_x_m(calibration)
    cell_x = length_m / cols
    masked = grid.copy()
    for col in range(cols):
        cell_center_x = (col + 0.5) * cell_x
        keep = cell_center_x < mid_x if half == "low" else cell_center_x > mid_x
        if not keep:
            masked[:, col] = 0.0
    return masked
