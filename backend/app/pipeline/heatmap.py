"""Player position heatmap: pitch grid, binning, smoothing, top-view render."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from backend.app.pipeline.movement_stats import trajectory_points_from_rows
from backend.app.pipeline.pitch_homography import (
    DEFAULT_PIXELS_PER_METER,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    PitchCalibration,
    is_badminton_court,
    is_on_pitch,
    render_badminton_court_template,
    render_pitch_template,
)
from backend.app.pipeline.sports.badminton_utils import (
    CourtSide,
    filter_positions_to_court_half,
    filter_rows_by_court_side,
    mask_grid_to_court_half,
    median_foot_x_px,
    metre_half_for_court_side,
)
from backend.app.schemas import Row

# ~2 m cells on a FIFA pitch (105 × 68 m)
DEFAULT_GRID_COLS = 53
DEFAULT_GRID_ROWS = 34
DEFAULT_SMOOTH_SIGMA = 1.5
HEATMAP_ALPHA = 0.55


@dataclass(frozen=True)
class HeatmapConfig:
    grid_cols: int = DEFAULT_GRID_COLS
    grid_rows: int = DEFAULT_GRID_ROWS
    smooth_sigma: float = DEFAULT_SMOOTH_SIGMA
    pixels_per_meter: int = DEFAULT_PIXELS_PER_METER
    alpha: float = HEATMAP_ALPHA


@dataclass(frozen=True)
class HeatmapZoneSummary:
    defensive_third_pct: float
    middle_third_pct: float
    attacking_third_pct: float
    left_pct: float
    center_pct: float
    right_pct: float
    central_pct: float
    wide_pct: float
    hottest_x_m: float | None
    hottest_y_m: float | None


@dataclass(frozen=True)
class HeatmapResult:
    grid_cols: int
    grid_rows: int
    pitch_length_m: float
    pitch_width_m: float
    sample_count: int
    image_png_base64: str
    zone_summary: HeatmapZoneSummary | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pct(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * count / total, 1)


def summarize_positions(
    positions: list[tuple[float, float]],
    *,
    length_m: float,
    width_m: float,
) -> HeatmapZoneSummary:
    """Zone occupancy from raw metre samples (not smoothed grid)."""
    n = len(positions)
    if n == 0:
        return HeatmapZoneSummary(
            defensive_third_pct=0.0,
            middle_third_pct=0.0,
            attacking_third_pct=0.0,
            left_pct=0.0,
            center_pct=0.0,
            right_pct=0.0,
            central_pct=0.0,
            wide_pct=0.0,
            hottest_x_m=None,
            hottest_y_m=None,
        )

    third_len = length_m / 3.0
    third_w = width_m / 3.0
    central_lo = width_m * 0.25
    central_hi = width_m * 0.75

    defensive = middle = attacking = 0
    left = center = right = 0
    central = wide = 0

    for x_m, y_m in positions:
        if x_m < third_len:
            defensive += 1
        elif x_m < 2.0 * third_len:
            middle += 1
        else:
            attacking += 1

        if y_m < third_w:
            left += 1
        elif y_m < 2.0 * third_w:
            center += 1
        else:
            right += 1

        if central_lo <= y_m <= central_hi:
            central += 1
        else:
            wide += 1

    return HeatmapZoneSummary(
        defensive_third_pct=_pct(defensive, n),
        middle_third_pct=_pct(middle, n),
        attacking_third_pct=_pct(attacking, n),
        left_pct=_pct(left, n),
        center_pct=_pct(center, n),
        right_pct=_pct(right, n),
        central_pct=_pct(central, n),
        wide_pct=_pct(wide, n),
        hottest_x_m=None,
        hottest_y_m=None,
    )


def hottest_zone_metres(
    grid: np.ndarray,
    *,
    length_m: float,
    width_m: float,
    grid_cols: int,
    grid_rows: int,
) -> tuple[float | None, float | None]:
    """Centre of the busiest grid cell in metres."""
    if grid.size == 0 or grid.max() <= 0:
        return None, None
    row, col = (int(i) for i in np.unravel_index(int(np.argmax(grid)), grid.shape))
    cell_x, cell_y = pitch_grid_shape(
        length_m, width_m, grid_cols=grid_cols, grid_rows=grid_rows
    )
    return round((col + 0.5) * cell_x, 1), round((row + 0.5) * cell_y, 1)


def build_zone_summary(
    positions: list[tuple[float, float]],
    grid: np.ndarray,
    *,
    length_m: float,
    width_m: float,
    grid_cols: int,
    grid_rows: int,
) -> HeatmapZoneSummary:
    summary = summarize_positions(positions, length_m=length_m, width_m=width_m)
    hot_x, hot_y = hottest_zone_metres(
        grid,
        length_m=length_m,
        width_m=width_m,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
    )
    return HeatmapZoneSummary(
        defensive_third_pct=summary.defensive_third_pct,
        middle_third_pct=summary.middle_third_pct,
        attacking_third_pct=summary.attacking_third_pct,
        left_pct=summary.left_pct,
        center_pct=summary.center_pct,
        right_pct=summary.right_pct,
        central_pct=summary.central_pct,
        wide_pct=summary.wide_pct,
        hottest_x_m=hot_x,
        hottest_y_m=hot_y,
    )


def pitch_grid_shape(
    length_m: float,
    width_m: float,
    *,
    grid_cols: int,
    grid_rows: int,
) -> tuple[float, float]:
    """Cell size in meters (x along length, y along width)."""
    if grid_cols < 1 or grid_rows < 1:
        raise ValueError("grid_cols and grid_rows must be at least 1.")
    return length_m / grid_cols, width_m / grid_rows


def meter_positions_from_rows(
    rows: list[Row] | list[dict[str, Any]],
    calibration: PitchCalibration,
    *,
    fps: float = 30.0,
    court_side: CourtSide | str | None = None,
) -> list[tuple[float, float]]:
    """Foot positions on the pitch in meters (x along length, y along width)."""
    scoped = rows
    image_probe_x: float | None = None
    if court_side in ("near", "far") and is_badminton_court(
        calibration.pitch_length_m, calibration.pitch_width_m
    ):
        scoped = filter_rows_by_court_side(rows, calibration, court_side)
        image_probe_x = median_foot_x_px(scoped, calibration)
    points = trajectory_points_from_rows(scoped, fps=fps, calibration=calibration)
    length_m = calibration.pitch_length_m
    width_m = calibration.pitch_width_m
    positions: list[tuple[float, float]] = []
    for _t, x_m, y_m in points:
        if is_on_pitch(x_m, y_m, length_m=length_m, width_m=width_m):
            positions.append((x_m, y_m))
    if court_side in ("near", "far"):
        positions = filter_positions_to_court_half(
            positions,
            calibration,
            court_side,
            image_probe_x=image_probe_x,
        )
    return positions


def bin_positions(
    positions: list[tuple[float, float]],
    *,
    length_m: float,
    width_m: float,
    grid_cols: int,
    grid_rows: int,
) -> np.ndarray:
    """Histogram player positions into a (grid_rows, grid_cols) count grid."""
    grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    if not positions:
        return grid

    xs = np.array([p[0] for p in positions], dtype=np.float64)
    ys = np.array([p[1] for p in positions], dtype=np.float64)

    # histogram2d: x bins along length, y bins along width
    counts, _, _ = np.histogram2d(
        xs,
        ys,
        bins=[grid_cols, grid_rows],
        range=[[0.0, length_m], [0.0, width_m]],
    )
    # histogram2d returns shape (grid_cols, grid_rows); transpose to row-major image
    return counts.T.astype(np.float32)


def smooth_grid(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smooth of occupancy counts."""
    if grid.size == 0 or sigma <= 0:
        return grid
    k = max(3, int(round(sigma * 4)) | 1)
    return cv2.GaussianBlur(grid, (k, k), sigmaX=sigma, sigmaY=sigma)


def grid_to_heatmap_rgba(grid: np.ndarray) -> np.ndarray:
    """Normalize smoothed counts to BGRA heat layer (transparent where zero)."""
    if grid.max() <= 0:
        return np.zeros((*grid.shape, 4), dtype=np.uint8)

    norm = grid / grid.max()
    heat_u8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    alpha = (norm * 255).astype(np.uint8)
    rgba = cv2.cvtColor(colored, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    return rgba


def render_heatmap(
    grid: np.ndarray,
    *,
    length_m: float,
    width_m: float,
    config: HeatmapConfig | None = None,
) -> np.ndarray:
    """Overlay smoothed heat grid on a top-down pitch template (BGR)."""
    cfg = config or HeatmapConfig()
    if is_badminton_court(length_m, width_m):
        template = render_badminton_court_template(
            length_m=length_m,
            width_m=width_m,
            pixels_per_meter=cfg.pixels_per_meter,
        )
    else:
        template = render_pitch_template(
            length_m=length_m,
            width_m=width_m,
            pixels_per_meter=cfg.pixels_per_meter,
        )
    h, w = template.shape[:2]

    if grid.max() <= 0:
        return template

    heat_small = grid_to_heatmap_rgba(grid)
    heat_full = cv2.resize(heat_small, (w, h), interpolation=cv2.INTER_LINEAR)

    mask = heat_full[:, :, 3:4].astype(np.float32) / 255.0
    alpha = cfg.alpha * mask
    base = template.astype(np.float32)
    overlay = heat_full[:, :, :3].astype(np.float32)
    blended = base * (1.0 - alpha) + overlay * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def encode_png_base64(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode heatmap PNG.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_heatmap(
    rows: list[Row] | list[dict[str, Any]],
    calibration: PitchCalibration,
    *,
    fps: float = 30.0,
    config: HeatmapConfig | None = None,
    court_side: CourtSide | str | None = None,
) -> tuple[HeatmapResult, np.ndarray]:
    """Full pipeline: rows → meter positions → grid → smooth → render."""
    cfg = config or HeatmapConfig()
    image_probe_x: float | None = None
    if court_side in ("near", "far") and is_badminton_court(
        calibration.pitch_length_m, calibration.pitch_width_m
    ):
        scoped = filter_rows_by_court_side(rows, calibration, court_side)
        image_probe_x = median_foot_x_px(scoped, calibration)
    positions = meter_positions_from_rows(
        rows, calibration, fps=fps, court_side=court_side
    )
    grid = bin_positions(
        positions,
        length_m=calibration.pitch_length_m,
        width_m=calibration.pitch_width_m,
        grid_cols=cfg.grid_cols,
        grid_rows=cfg.grid_rows,
    )
    if (
        court_side in ("near", "far")
        and positions
        and is_badminton_court(calibration.pitch_length_m, calibration.pitch_width_m)
    ):
        half = metre_half_for_court_side(
            positions, calibration, court_side, image_probe_x=image_probe_x
        )
        grid = mask_grid_to_court_half(grid, calibration, half)
    smoothed = smooth_grid(grid, cfg.smooth_sigma)
    image = render_heatmap(
        smoothed,
        length_m=calibration.pitch_length_m,
        width_m=calibration.pitch_width_m,
        config=cfg,
    )
    zone_summary = build_zone_summary(
        positions,
        grid,
        length_m=calibration.pitch_length_m,
        width_m=calibration.pitch_width_m,
        grid_cols=cfg.grid_cols,
        grid_rows=cfg.grid_rows,
    )
    result = HeatmapResult(
        grid_cols=cfg.grid_cols,
        grid_rows=cfg.grid_rows,
        pitch_length_m=calibration.pitch_length_m,
        pitch_width_m=calibration.pitch_width_m,
        sample_count=len(positions),
        image_png_base64=encode_png_base64(image),
        zone_summary=zone_summary,
    )
    return result, image


def build_heatmap_from_rows(
    rows: list[Row] | list[dict[str, Any]],
    calibration: PitchCalibration,
    *,
    fps: float = 30.0,
    config: HeatmapConfig | None = None,
    court_side: CourtSide | str | None = None,
) -> HeatmapResult:
    result, _ = build_heatmap(
        rows, calibration, fps=fps, config=config, court_side=court_side
    )
    return result


def metre_positions_from_points(
    points: list[tuple[float, float]] | list[tuple[float, float, float]],
    *,
    length_m: float = PITCH_LENGTH_M,
    width_m: float = PITCH_WIDTH_M,
) -> list[tuple[float, float]]:
    """Filter (x_m, y_m) to in-pitch bounds."""
    positions: list[tuple[float, float]] = []
    for p in points:
        if len(p) >= 3:
            x_m, y_m = float(p[1]), float(p[2])
        else:
            x_m, y_m = float(p[0]), float(p[1])
        if is_on_pitch(x_m, y_m, length_m=length_m, width_m=width_m):
            positions.append((x_m, y_m))
    return positions


def build_heatmap_from_metre_points(
    positions: list[tuple[float, float]],
    *,
    length_m: float = PITCH_LENGTH_M,
    width_m: float = PITCH_WIDTH_M,
    config: HeatmapConfig | None = None,
) -> HeatmapResult:
    """Heatmap from precomputed metre positions."""
    cfg = config or HeatmapConfig()
    filtered = metre_positions_from_points(
        positions, length_m=length_m, width_m=width_m
    )
    grid = bin_positions(
        filtered,
        length_m=length_m,
        width_m=width_m,
        grid_cols=cfg.grid_cols,
        grid_rows=cfg.grid_rows,
    )
    smoothed = smooth_grid(grid, cfg.smooth_sigma)
    image = render_heatmap(
        smoothed,
        length_m=length_m,
        width_m=width_m,
        config=cfg,
    )
    zone_summary = build_zone_summary(
        filtered,
        grid,
        length_m=length_m,
        width_m=width_m,
        grid_cols=cfg.grid_cols,
        grid_rows=cfg.grid_rows,
    )
    return HeatmapResult(
        grid_cols=cfg.grid_cols,
        grid_rows=cfg.grid_rows,
        pitch_length_m=length_m,
        pitch_width_m=width_m,
        sample_count=len(filtered),
        image_png_base64=encode_png_base64(image),
        zone_summary=zone_summary,
    )
