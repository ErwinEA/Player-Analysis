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
    is_on_pitch,
    render_pitch_template,
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
class HeatmapResult:
    grid_cols: int
    grid_rows: int
    pitch_length_m: float
    pitch_width_m: float
    sample_count: int
    image_png_base64: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
) -> list[tuple[float, float]]:
    """Foot positions on the pitch in meters (x along length, y along width)."""
    points = trajectory_points_from_rows(rows, fps=fps, calibration=calibration)
    length_m = calibration.pitch_length_m
    width_m = calibration.pitch_width_m
    positions: list[tuple[float, float]] = []
    for _t, x_m, y_m in points:
        if is_on_pitch(x_m, y_m, length_m=length_m, width_m=width_m):
            positions.append((x_m, y_m))
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
) -> tuple[HeatmapResult, np.ndarray]:
    """Full pipeline: rows → meter positions → grid → smooth → render."""
    cfg = config or HeatmapConfig()
    positions = meter_positions_from_rows(rows, calibration, fps=fps)
    grid = bin_positions(
        positions,
        length_m=calibration.pitch_length_m,
        width_m=calibration.pitch_width_m,
        grid_cols=cfg.grid_cols,
        grid_rows=cfg.grid_rows,
    )
    smoothed = smooth_grid(grid, cfg.smooth_sigma)
    image = render_heatmap(
        smoothed,
        length_m=calibration.pitch_length_m,
        width_m=calibration.pitch_width_m,
        config=cfg,
    )
    result = HeatmapResult(
        grid_cols=cfg.grid_cols,
        grid_rows=cfg.grid_rows,
        pitch_length_m=calibration.pitch_length_m,
        pitch_width_m=calibration.pitch_width_m,
        sample_count=len(positions),
        image_png_base64=encode_png_base64(image),
    )
    return result, image


def build_heatmap_from_rows(
    rows: list[Row] | list[dict[str, Any]],
    calibration: PitchCalibration,
    *,
    fps: float = 30.0,
    config: HeatmapConfig | None = None,
) -> HeatmapResult:
    result, _ = build_heatmap(rows, calibration, fps=fps, config=config)
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
    return HeatmapResult(
        grid_cols=cfg.grid_cols,
        grid_rows=cfg.grid_rows,
        pitch_length_m=length_m,
        pitch_width_m=width_m,
        sample_count=len(filtered),
        image_png_base64=encode_png_base64(image),
    )
