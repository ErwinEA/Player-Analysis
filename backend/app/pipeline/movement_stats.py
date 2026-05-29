"""Movement statistics from locked-player trajectory rows."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from backend.app.pipeline.pitch_homography import PitchCalibration, is_on_pitch
from backend.app.schemas import Row

# ~25 km/h — common broadcast-analytics sprint threshold
SPRINT_SPEED_M_S = 7.0
MIN_SPRINT_DURATION_S = 1.0
MAX_SEGMENT_GAP_S = 2.0
MAX_SPEED_M_S = 12.0
MIN_DT_S = 1e-6


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
    x1, _y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, y2


def trajectory_points_from_metres(
    points: list[tuple[float, float, float]] | list[Any],
    *,
    fps: float = 30.0,
) -> list[tuple[float, float, float]]:
    """Accept (t, x_m, y_m) or objects with t/frame/x_m/y_m attributes."""
    out: list[tuple[float, float, float]] = []
    for p in points:
        if hasattr(p, "t") and hasattr(p, "x_m") and hasattr(p, "y_m"):
            t = float(p.t)
            x_m = float(p.x_m)
            y_m = float(p.y_m)
        elif isinstance(p, (list, tuple)) and len(p) >= 3:
            t, x_m, y_m = float(p[0]), float(p[1]), float(p[2])
        elif isinstance(p, dict):
            t = p.get("t")
            if t is None:
                frame = int(p.get("frame", 0))
                t = frame / fps if fps > 0 else 0.0
            x_m = float(p["x_m"])
            y_m = float(p["y_m"])
        else:
            continue
        out.append((t, x_m, y_m))
    return sorted(out, key=lambda pt: pt[0])


def compute_movement_stats_from_metres(
    points: list[tuple[float, float, float]] | list[Any],
    *,
    fps: float = 30.0,
    config: MovementStatsConfig | None = None,
) -> MovementStats:
    traj = trajectory_points_from_metres(points, fps=fps)
    return compute_movement_stats(traj, config=config, units="meters")


def trajectory_points_from_rows(
    rows: list[Row] | list[dict[str, Any]],
    *,
    fps: float = 30.0,
    calibration: PitchCalibration | None = None,
) -> list[tuple[float, float, float]]:
    dict_rows: list[dict[str, Any]] = [
        r.model_dump() if hasattr(r, "model_dump") else r for r in rows
    ]
    to_meters = calibration.pixel_to_meters if calibration is not None else None
    points: list[tuple[float, float, float]] = []
    for row in sorted(dict_rows, key=lambda r: int(r["frame"])):
        bbox = row.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        foot_x = row.get("foot_x")
        foot_y = row.get("foot_y")
        if (
            foot_x is not None
            and foot_y is not None
            and calibration is not None
        ):
            fw, fh = calibration.image_size
            px = float(foot_x) * fw
            py = float(foot_y) * fh
        else:
            px, py = foot_position_pixels([float(v) for v in bbox])
        if to_meters is not None:
            try:
                px, py = to_meters(px, py)
            except ValueError:
                continue
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
    segment_speeds: list[tuple[float, float, float]] = []

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


def _filter_trajectory_to_pitch(
    points: list[tuple[float, float, float]],
    calibration: PitchCalibration,
) -> list[tuple[float, float, float]]:
    length_m = calibration.pitch_length_m
    width_m = calibration.pitch_width_m
    return [
        pt
        for pt in points
        if is_on_pitch(pt[1], pt[2], length_m=length_m, width_m=width_m)
    ]


def compute_movement_stats_from_rows(
    rows: list[Row] | list[dict[str, Any]],
    *,
    fps: float = 30.0,
    calibration: PitchCalibration | None = None,
    config: MovementStatsConfig | None = None,
) -> MovementStats:
    units = "meters" if calibration is not None else "pixels"
    points = trajectory_points_from_rows(rows, fps=fps, calibration=calibration)
    if calibration is not None:
        points = _filter_trajectory_to_pitch(points, calibration)
    return compute_movement_stats(points, config=config, units=units)
