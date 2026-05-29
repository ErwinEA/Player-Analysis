"""Movement statistics from a sparse player trajectory (bbox time series)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

# ~25 km/h — common broadcast-analytics sprint threshold
SPRINT_SPEED_M_S = 7.0
MIN_SPRINT_DURATION_S = 1.0

# Ignore gaps (re-lock / dropped frames) and cap detection jitter
MAX_SEGMENT_GAP_S = 2.0
MAX_SPEED_M_S = 12.0

MIN_DT_S = 1e-6


class _MeterMapper(Protocol):
    def __call__(self, x: float, y: float) -> tuple[float, float]: ...


@dataclass(frozen=True)
class MovementStatsConfig:
    sprint_speed_m_s: float = SPRINT_SPEED_M_S
    min_sprint_duration_s: float = MIN_SPRINT_DURATION_S
    max_segment_gap_s: float = MAX_SEGMENT_GAP_S
    max_speed_m_s: float = MAX_SPEED_M_S


@dataclass(frozen=True)
class MovementStats:
    distance_m: float
    max_speed_m_s: float
    avg_speed_m_s: float
    sprint_count: int
    tracked_duration_s: float
    sample_count: int
    units: str = "meters"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def foot_position_pixels(bbox: list[float]) -> tuple[float, float]:
    """Bottom-center of bbox in image pixels (tracking foot point)."""
    x1, _y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, y2


def _pixel_to_meters_from_homography(H: list[list[float]], x: float, y: float) -> tuple[float, float]:
    w = H[0][0] * x + H[0][1] * y + H[0][2]
    h = H[1][0] * x + H[1][1] * y + H[1][2]
    z = H[2][0] * x + H[2][1] * y + H[2][2]
    if abs(z) < 1e-9:
        raise ValueError(f"Point ({x}, {y}) maps to infinity under homography.")
    return w / z, h / z


def load_calibration_mapper(path: Path) -> _MeterMapper:
    data = json.loads(path.read_text(encoding="utf-8"))
    H = data["homography"]

    def mapper(x: float, y: float) -> tuple[float, float]:
        return _pixel_to_meters_from_homography(H, x, y)

    return mapper


def trajectory_points_from_rows(
    rows: list[dict[str, Any]],
    *,
    fps: float = 30.0,
    to_meters: _MeterMapper | None = None,
) -> list[tuple[float, float, float]]:
    """Return sorted (t, x, y) samples; y is meters if to_meters is set else pixels."""
    if not rows:
        return []

    sorted_rows = sorted(rows, key=lambda r: int(r["frame"]))
    points: list[tuple[float, float, float]] = []
    for row in sorted_rows:
        bbox = row.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        px, py = foot_position_pixels([float(v) for v in bbox])
        if to_meters is not None:
            px, py = to_meters(px, py)
        t = row.get("t")
        if t is None:
            t = int(row["frame"]) / fps if fps > 0 else 0.0
        points.append((float(t), float(px), float(py)))
    return points


def compute_movement_stats(
    points: list[tuple[float, float, float]],
    *,
    config: MovementStatsConfig | None = None,
    units: str = "meters",
) -> MovementStats:
    """Compute distance, speeds, and sprint count from (t, x, y) samples."""
    cfg = config or MovementStatsConfig()
    empty = MovementStats(
        distance_m=0.0,
        max_speed_m_s=0.0,
        avg_speed_m_s=0.0,
        sprint_count=0,
        tracked_duration_s=0.0,
        sample_count=len(points),
        units=units,
    )
    if len(points) < 2:
        return empty

    total_distance = 0.0
    active_time = 0.0
    max_speed = 0.0
    segment_speeds: list[tuple[float, float, float]] = []  # (t_mid, speed, dt)

    for (t0, x0, y0), (t1, x1, y1) in zip(points, points[1:]):
        dt = t1 - t0
        if dt < MIN_DT_S:
            continue
        if dt > cfg.max_segment_gap_s:
            continue
        dist = math.hypot(x1 - x0, y1 - y0)
        speed = min(dist / dt, cfg.max_speed_m_s)
        total_distance += dist
        active_time += dt
        max_speed = max(max_speed, speed)
        segment_speeds.append(((t0 + t1) / 2.0, speed, dt))

    tracked_duration = points[-1][0] - points[0][0]
    avg_speed = total_distance / active_time if active_time > MIN_DT_S else 0.0

    sprint_count = 0
    if units == "meters":
        sprint_count = _count_sprints(
            segment_speeds,
            threshold=cfg.sprint_speed_m_s,
            min_duration=cfg.min_sprint_duration_s,
        )

    return MovementStats(
        distance_m=round(total_distance, 3),
        max_speed_m_s=round(max_speed, 3),
        avg_speed_m_s=round(avg_speed, 3),
        sprint_count=sprint_count,
        tracked_duration_s=round(tracked_duration, 3),
        sample_count=len(points),
        units=units,
    )


def _count_sprints(
    segments: list[tuple[float, float, float]],
    *,
    threshold: float,
    min_duration: float,
) -> int:
    """Count sprint episodes where speed stays at or above threshold for min_duration."""
    count = 0
    in_sprint = False
    sprint_start_t: float | None = None

    for t_mid, speed, dt in segments:
        t_start = t_mid - dt / 2.0
        if speed >= threshold:
            if not in_sprint:
                in_sprint = True
                sprint_start_t = t_start
        elif in_sprint and sprint_start_t is not None:
            if t_start - sprint_start_t >= min_duration:
                count += 1
            in_sprint = False
            sprint_start_t = None

    if in_sprint and sprint_start_t is not None and segments:
        last_t, _, last_dt = segments[-1]
        sprint_end = last_t + last_dt / 2.0
        if sprint_end - sprint_start_t >= min_duration:
            count += 1

    return count


def compute_movement_stats_from_rows(
    rows: list[dict[str, Any]],
    *,
    fps: float = 30.0,
    calibration_path: Path | None = None,
    to_meters: Callable[[float, float], tuple[float, float]] | None = None,
    config: MovementStatsConfig | None = None,
) -> MovementStats:
    """Convenience: rows (identify / pipeline target.rows) → MovementStats."""
    mapper = to_meters
    units = "meters"
    if mapper is None and calibration_path is not None:
        mapper = load_calibration_mapper(calibration_path)
    if mapper is None:
        units = "pixels"
    points = trajectory_points_from_rows(rows, fps=fps, to_meters=mapper)
    return compute_movement_stats(points, config=config, units=units)
