"""Pitch homography: manual corner calibration, pixel↔meter transforms, top-view template."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# FIFA standard pitch (meters)
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# Corner order: top-left, top-right, bottom-right, bottom-left (image y-down)
CORNER_LABELS = ("top_left", "top_right", "bottom_right", "bottom_left")

BOUNDARY_POINT_COUNT = 10
MIN_BOUNDARY_POINTS = 4
MAX_BOUNDARY_POINTS = 20
PROBE_COUNT_REQUIRED = 3
PROBE_TOTAL = 5
# Indices of the four pitch corners within the 10-point clockwise outline
BOUNDARY_CORNER_INDICES = (0, 3, 6, 9)
# Clockwise from top-left around the visible pitch outline
BOUNDARY_LABELS = (
    "Top left corner",
    "Top line (left)",
    "Top line (center)",
    "Top right corner",
    "Right side (upper)",
    "Right side (lower)",
    "Bottom right corner",
    "Bottom line (right)",
    "Bottom line (left)",
    "Bottom left corner",
)

DEFAULT_PIXELS_PER_METER = 10

_CALIBRATION_PROBE_FRACTIONS = (
    (0.5, 0.5),
    (0.25, 0.55),
    (0.75, 0.55),
    (0.5, 0.72),
    (0.5, 0.35),
)


@dataclass(frozen=True)
class CalibrationDiagnostics:
    """Pre-save homography quality metrics (outline mode)."""

    probe_count: int
    probe_total: int
    probe_details: tuple[dict[str, Any], ...]
    coverage_pct: float
    confidence: float
    warnings: tuple[str, ...]
    fitted_quad: tuple[list[float], ...]
    mode: str = "outline"


@dataclass(frozen=True)
class PitchCalibration:
    """Homography from image pixels to pitch meters (Z=0 plane)."""

    name: str
    video_path: str | None
    frame_index: int
    image_size: tuple[int, int]  # width, height
    image_corners: list[list[float]]  # [[x,y], ...] TL, TR, BR, BL (warp quad)
    pitch_length_m: float
    pitch_width_m: float
    homography: list[list[float]]  # 3x3 image -> meters
    homography_inv: list[list[float]]  # 3x3 meters -> image
    image_boundary_points: list[list[float]] | None = None  # outline (optional)
    mode: str = "outline"
    confidence: float | None = None
    coverage_pct: float | None = None

    @property
    def H(self) -> np.ndarray:
        return np.array(self.homography, dtype=np.float64)

    @property
    def H_inv(self) -> np.ndarray:
        return np.array(self.homography_inv, dtype=np.float64)

    def pixel_to_meters(self, x: float, y: float) -> tuple[float, float]:
        pt = np.array([x, y, 1.0], dtype=np.float64)
        mapped = self.H @ pt
        if abs(mapped[2]) < 1e-9:
            raise ValueError("Point maps to infinity (likely outside pitch plane).")
        mx, my = mapped[0] / mapped[2], mapped[1] / mapped[2]
        return float(mx), float(my)

    def meters_to_pixel(self, x_m: float, y_m: float) -> tuple[float, float]:
        pt = np.array([x_m, y_m, 1.0], dtype=np.float64)
        mapped = self.H_inv @ pt
        if abs(mapped[2]) < 1e-9:
            raise ValueError("Meter point maps to infinity.")
        px, py = mapped[0] / mapped[2], mapped[1] / mapped[2]
        return float(px), float(py)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "video_path": self.video_path,
            "frame_index": self.frame_index,
            "image_size": list(self.image_size),
            "image_corners": self.image_corners,
            "image_boundary_points": self.image_boundary_points,
            "corner_labels": list(CORNER_LABELS),
            "boundary_labels": list(BOUNDARY_LABELS),
            "pitch_length_m": self.pitch_length_m,
            "pitch_width_m": self.pitch_width_m,
            "homography": self.homography,
            "homography_inv": self.homography_inv,
            "mode": self.mode,
            "confidence": self.confidence,
            "coverage_pct": self.coverage_pct,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PitchCalibration:
        image_size = (int(data["image_size"][0]), int(data["image_size"][1]))
        length_m = float(data.get("pitch_length_m", PITCH_LENGTH_M))
        width_m = float(data.get("pitch_width_m", PITCH_WIDTH_M))
        boundary_raw = data.get("image_boundary_points")
        corners_raw = data.get("image_corners")
        w, h = image_size
        if boundary_raw and len(boundary_raw) >= MIN_BOUNDARY_POINTS:
            boundary = [list(map(float, p)) for p in boundary_raw]
            if len(boundary) > MAX_BOUNDARY_POINTS:
                raise ValueError(
                    f"At most {MAX_BOUNDARY_POINTS} boundary points allowed, got {len(boundary)}."
                )
            corners = decagon_to_quad(boundary, width=w, height=h)
        elif corners_raw and len(corners_raw) == 4:
            corners = [list(map(float, p)) for p in corners_raw]
            boundary = None
        else:
            raise ValueError(
                "Calibration JSON needs image_boundary_points (4–20) or 4 image_corners."
            )
        H, H_inv = compute_homography(
            corners,
            width=w,
            height=h,
            length_m=length_m,
            width_m=width_m,
        )
        return cls(
            name=str(data["name"]),
            video_path=data.get("video_path"),
            frame_index=int(data.get("frame_index", 0)),
            image_size=image_size,
            image_corners=corners,
            image_boundary_points=boundary,
            pitch_length_m=length_m,
            pitch_width_m=width_m,
            homography=H.tolist(),
            homography_inv=H_inv.tolist(),
            mode=str(data.get("mode", "outline")),
            confidence=(
                float(data["confidence"]) if data.get("confidence") is not None else None
            ),
            coverage_pct=(
                float(data["coverage_pct"])
                if data.get("coverage_pct") is not None
                else None
            ),
        )


def is_on_pitch(
    x_m: float,
    y_m: float,
    *,
    length_m: float = PITCH_LENGTH_M,
    width_m: float = PITCH_WIDTH_M,
) -> bool:
    return 0.0 <= x_m <= length_m and 0.0 <= y_m <= width_m


def _calibration_probe_pixels(cal: PitchCalibration) -> tuple[tuple[float, float], ...]:
    w, h = cal.image_size
    return tuple((w * fx, h * fy) for fx, fy in _CALIBRATION_PROBE_FRACTIONS)


def probe_calibration(cal: PitchCalibration) -> tuple[int, list[dict[str, Any]]]:
    """Sample frame probes; return count on-pitch and per-probe details."""
    details: list[dict[str, Any]] = []
    in_bounds = 0
    for px, py in _calibration_probe_pixels(cal):
        entry: dict[str, Any] = {
            "px": round(px, 1),
            "py": round(py, 1),
            "on_pitch": False,
        }
        try:
            x_m, y_m = cal.pixel_to_meters(px, py)
            entry["x_m"] = round(x_m, 2)
            entry["y_m"] = round(y_m, 2)
            on = is_on_pitch(
                x_m,
                y_m,
                length_m=cal.pitch_length_m,
                width_m=cal.pitch_width_m,
            )
            entry["on_pitch"] = on
            if on:
                in_bounds += 1
        except ValueError:
            entry["error"] = "maps_to_infinity"
        details.append(entry)
    return in_bounds, details


def validate_calibration_maps_to_pitch(cal: PitchCalibration) -> None:
    """Reject calibrations whose homography does not map the frame onto the pitch."""
    in_bounds, _ = probe_calibration(cal)
    if in_bounds < PROBE_COUNT_REQUIRED:
        raise ValueError(
            "Calibration does not map the video frame onto the pitch. "
            "Re-mark boundary points on a clear wide shot around the visible pitch outline."
        )


def compute_grid_coverage(
    cal: PitchCalibration,
    *,
    grid_x: int = 5,
    grid_y: int = 4,
) -> float:
    """Fraction of pitch grid points that project into the image frame."""
    w, h = cal.image_size
    length_m = cal.pitch_length_m
    width_m = cal.pitch_width_m
    xs = np.linspace(0.0, length_m, grid_x) if grid_x > 1 else np.array([0.0])
    ys = np.linspace(0.0, width_m, grid_y) if grid_y > 1 else np.array([0.0])
    valid = 0
    total = 0
    for x_m in xs:
        for y_m in ys:
            total += 1
            try:
                px, py = cal.meters_to_pixel(float(x_m), float(y_m))
            except ValueError:
                continue
            if 0.0 <= px < w and 0.0 <= py < h:
                valid += 1
    return valid / total if total else 0.0


def compute_calibration_confidence(
    *,
    probe_count: int,
    coverage_pct: float,
    probe_total: int = PROBE_TOTAL,
) -> tuple[float, list[str]]:
    """Score calibration quality 0–1 with human-readable warnings."""
    warnings: list[str] = []
    confidence = 0.5
    if probe_count >= PROBE_COUNT_REQUIRED:
        confidence += 0.3
    else:
        warnings.append(
            f"Only {probe_count}/{probe_total} frame probes map onto the pitch; "
            "try a wider shot or adjust boundary points."
        )
        confidence = min(confidence, 0.5)
    if coverage_pct >= 0.7:
        confidence += 0.2
    elif coverage_pct < 0.6:
        warnings.append(
            f"Pitch grid coverage is {coverage_pct * 100:.0f}% (<60%); "
            "some pitch regions may not map correctly in this frame."
        )
    if probe_count < PROBE_COUNT_REQUIRED:
        confidence = min(confidence, 0.2)
    else:
        confidence = max(0.2, min(1.0, confidence))
    return round(confidence, 3), warnings


def world_corners_meters(
    length_m: float = PITCH_LENGTH_M,
    width_m: float = PITCH_WIDTH_M,
) -> np.ndarray:
    """Pitch rectangle in meters: TL, TR, BR, BL."""
    return np.float32(
        [
            [0.0, 0.0],
            [length_m, 0.0],
            [length_m, width_m],
            [0.0, width_m],
        ]
    )


def _validate_corners(image_corners: np.ndarray, width: int, height: int) -> None:
    if image_corners.shape != (4, 2):
        raise ValueError("Exactly 4 corner points required (TL, TR, BR, BL).")
    # Broadcast angles often need extrapolated corners slightly off-frame; allow margin.
    margin_x = width * 0.15
    margin_y = height * 0.15
    for x, y in image_corners:
        if not (-margin_x <= x <= width + margin_x and -margin_y <= y <= height + margin_y):
            raise ValueError(
                f"Corner ({x:.1f}, {y:.1f}) too far outside image bounds {width}x{height}."
            )
    # Reject only truly degenerate input (near-collinear / coincident points).
    area = _quad_area(image_corners)
    min_dim = min(width, height)
    if area < (min_dim * 0.01) ** 2:
        raise ValueError(
            "Corner points are nearly collinear or coincident — spread them across the visible pitch."
        )


def _order_corners_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL for perspective warp."""
    p = pts.reshape(4, 2).astype(np.float64)
    by_sum = p[:, 0] + p[:, 1]
    by_diff = p[:, 0] - p[:, 1]
    used: set[int] = set()
    picks: list[int] = []
    for key in (
        int(np.argmin(by_sum)),
        int(np.argmax(by_diff)),
        int(np.argmax(by_sum)),
        int(np.argmin(by_diff)),
    ):
        if key not in used:
            used.add(key)
            picks.append(key)
    for i in range(4):
        if i not in used:
            picks.append(i)
    ordered = p[picks[:4]]
    return np.float32(ordered)


def _validate_boundary_points(
    points: np.ndarray, width: int, height: int
) -> None:
    margin_x = width * 0.15
    margin_y = height * 0.15
    for x, y in points:
        if not (-margin_x <= x <= width + margin_x and -margin_y <= y <= height + margin_y):
            raise ValueError(
                f"Boundary point ({x:.1f}, {y:.1f}) too far outside image bounds {width}x{height}."
            )


def decagon_to_quad(
    boundary_points: list[list[float]] | np.ndarray,
    *,
    width: int | None = None,
    height: int | None = None,
) -> list[list[float]]:
    """Fit a perspective quad from 4 or 10 clockwise boundary points on the pitch outline."""
    pts = np.asarray(boundary_points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < MIN_BOUNDARY_POINTS:
        raise ValueError(
            f"Need at least {MIN_BOUNDARY_POINTS} boundary points, got {pts.shape[0]}."
        )
    if pts.shape[0] > MAX_BOUNDARY_POINTS:
        raise ValueError(
            f"At most {MAX_BOUNDARY_POINTS} boundary points allowed, got {pts.shape[0]}."
        )
    if width is not None and height is not None:
        _validate_boundary_points(pts, width, height)

    if pts.shape[0] == BOUNDARY_POINT_COUNT:
        corner_pts = pts[list(BOUNDARY_CORNER_INDICES)]
        ordered = _order_corners_tl_tr_br_bl(corner_pts)
        return [[float(x), float(y)] for x, y in ordered]

    if pts.shape[0] == 4:
        # Legacy saves: corners already in TL, TR, BR, BL click order.
        return [[float(x), float(y)] for x, y in pts]

    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    n = len(ordered)

    quad = np.zeros((4, 2), dtype=np.float64)
    for i in range(4):
        start = int(round(i * n / 4))
        end = int(round((i + 1) * n / 4))
        if end <= start:
            end = min(start + 1, n)
        arc = ordered[start:end]
        dists = np.sum((arc - center) ** 2, axis=1)
        quad[i] = arc[int(np.argmax(dists))]

    ordered_quad = _order_corners_tl_tr_br_bl(quad)
    return [[float(x), float(y)] for x, y in ordered_quad]


def _quad_area(pts: np.ndarray) -> float:
    """Shoelace area; works for any simple quad, not just convex."""
    x = pts[:, 0]
    y = pts[:, 1]
    return float(
        abs(
            x[0] * y[1]
            + x[1] * y[2]
            + x[2] * y[3]
            + x[3] * y[0]
            - y[0] * x[1]
            - y[1] * x[2]
            - y[2] * x[3]
            - y[3] * x[0]
        )
        / 2.0
    )


def compute_homography(
    image_corners: list[list[float]] | np.ndarray,
    *,
    width: int,
    height: int,
    length_m: float = PITCH_LENGTH_M,
    width_m: float = PITCH_WIDTH_M,
) -> tuple[np.ndarray, np.ndarray]:
    src = np.float32(image_corners).reshape(4, 2)
    _validate_corners(src, width, height)
    dst = world_corners_meters(length_m, width_m)
    # getPerspectiveTransform accepts any 4 non-degenerate correspondences (no convex requirement).
    H = cv2.getPerspectiveTransform(src, dst)
    H_inv = cv2.getPerspectiveTransform(dst, src)
    if H is None or H_inv is None:
        raise ValueError("Could not compute homography from corner points.")
    return H, H_inv


def build_calibration(
    name: str,
    image_corners: list[list[float]] | None = None,
    *,
    image_boundary_points: list[list[float]] | None = None,
    video_path: str | None = None,
    frame_index: int = 0,
    image_size: tuple[int, int],
    length_m: float = PITCH_LENGTH_M,
    width_m: float = PITCH_WIDTH_M,
    mode: str = "outline",
    confidence: float | None = None,
    coverage_pct: float | None = None,
) -> PitchCalibration:
    w, h = image_size
    if image_boundary_points is not None:
        n_pts = len(image_boundary_points)
        if n_pts < MIN_BOUNDARY_POINTS or n_pts > MAX_BOUNDARY_POINTS:
            raise ValueError(
                f"Boundary point count must be {MIN_BOUNDARY_POINTS}–{MAX_BOUNDARY_POINTS}, "
                f"got {n_pts}."
            )
        boundary = [[float(x), float(y)] for x, y in image_boundary_points]
        corners = decagon_to_quad(boundary, width=w, height=h)
    elif image_corners is not None:
        corners = decagon_to_quad(image_corners, width=w, height=h)
        boundary = (
            [[float(x), float(y)] for x, y in image_corners]
            if len(image_corners) >= MIN_BOUNDARY_POINTS
            else None
        )
    else:
        raise ValueError("Provide image_boundary_points or image_corners.")
    H, H_inv = compute_homography(
        corners, width=w, height=h, length_m=length_m, width_m=width_m
    )
    return PitchCalibration(
        name=name,
        video_path=video_path,
        frame_index=frame_index,
        image_size=image_size,
        image_corners=corners,
        image_boundary_points=boundary,
        pitch_length_m=length_m,
        pitch_width_m=width_m,
        homography=H.tolist(),
        homography_inv=H_inv.tolist(),
        mode=mode,
        confidence=confidence,
        coverage_pct=coverage_pct,
    )


def run_calibration_diagnostics(cal: PitchCalibration) -> CalibrationDiagnostics:
    """Compute probe, coverage, and confidence metrics without raising on probe failure."""
    probe_count, probe_details = probe_calibration(cal)
    coverage = compute_grid_coverage(cal)
    confidence, warnings = compute_calibration_confidence(
        probe_count=probe_count,
        coverage_pct=coverage,
    )
    return CalibrationDiagnostics(
        probe_count=probe_count,
        probe_total=PROBE_TOTAL,
        probe_details=tuple(probe_details),
        coverage_pct=round(coverage, 4),
        confidence=confidence,
        warnings=tuple(warnings),
        fitted_quad=tuple(list(c) for c in cal.image_corners),
        mode=cal.mode,
    )


def build_calibration_with_diagnostics(
    name: str,
    image_corners: list[list[float]] | None = None,
    *,
    image_boundary_points: list[list[float]] | None = None,
    video_path: str | None = None,
    frame_index: int = 0,
    image_size: tuple[int, int],
    length_m: float = PITCH_LENGTH_M,
    width_m: float = PITCH_WIDTH_M,
) -> tuple[PitchCalibration, CalibrationDiagnostics]:
    """Build calibration and attach diagnostic metadata (does not raise on probe failure)."""
    cal = build_calibration(
        name,
        image_corners,
        image_boundary_points=image_boundary_points,
        video_path=video_path,
        frame_index=frame_index,
        image_size=image_size,
        length_m=length_m,
        width_m=width_m,
    )
    diag = run_calibration_diagnostics(cal)
    cal = PitchCalibration(
        name=cal.name,
        video_path=cal.video_path,
        frame_index=cal.frame_index,
        image_size=cal.image_size,
        image_corners=cal.image_corners,
        pitch_length_m=cal.pitch_length_m,
        pitch_width_m=cal.pitch_width_m,
        homography=cal.homography,
        homography_inv=cal.homography_inv,
        image_boundary_points=cal.image_boundary_points,
        mode=diag.mode,
        confidence=diag.confidence,
        coverage_pct=diag.coverage_pct,
    )
    return cal, diag


def save_calibration(cal: PitchCalibration, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal.to_dict(), indent=2), encoding="utf-8")


def load_calibration(path: Path) -> PitchCalibration:
    data = json.loads(path.read_text(encoding="utf-8"))
    return PitchCalibration.from_dict(data)


def default_calibration_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "pitch_calibration"


def render_pitch_template(
    *,
    length_m: float = PITCH_LENGTH_M,
    width_m: float = PITCH_WIDTH_M,
    pixels_per_meter: int = DEFAULT_PIXELS_PER_METER,
) -> np.ndarray:
    """Draw a 2D top-down FIFA pitch diagram (BGR) for the heatmap UI."""
    w = int(round(length_m * pixels_per_meter))
    h = int(round(width_m * pixels_per_meter))
    ppm = float(pixels_per_meter)

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (34, 120, 34)  # grass green

    white = (255, 255, 255)
    line = max(1, int(round(ppm * 0.12)))

    def m(x: float, y: float) -> tuple[int, int]:
        return int(round(x * ppm)), int(round(y * ppm))

    cv2.rectangle(img, m(0, 0), m(length_m, width_m), white, line)

    # Halfway line
    cv2.line(img, m(length_m / 2, 0), m(length_m / 2, width_m), white, line)

    # Center circle (9.15 m radius)
    center = m(length_m / 2, width_m / 2)
    cv2.circle(img, center, int(round(9.15 * ppm)), white, line)
    cv2.circle(img, center, max(2, line), white, -1)

    # Penalty areas (16.5 x 40.32)
    pa_depth = 16.5
    pa_width = 40.32
    pa_top = (width_m - pa_width) / 2
    cv2.rectangle(img, m(0, pa_top), m(pa_depth, pa_top + pa_width), white, line)
    cv2.rectangle(
        img,
        m(length_m - pa_depth, pa_top),
        m(length_m, pa_top + pa_width),
        white,
        line,
    )

    # Goal areas (5.5 x 18.32)
    ga_depth = 5.5
    ga_width = 18.32
    ga_top = (width_m - ga_width) / 2
    cv2.rectangle(img, m(0, ga_top), m(ga_depth, ga_top + ga_width), white, line)
    cv2.rectangle(
        img,
        m(length_m - ga_depth, ga_top),
        m(length_m, ga_top + ga_width),
        white,
        line,
    )

    # Penalty spots (11 m from goal line)
    spot_y = width_m / 2
    cv2.circle(img, m(11.0, spot_y), max(2, line), white, -1)
    cv2.circle(img, m(length_m - 11.0, spot_y), max(2, line), white, -1)

    return img


def badminton_court_asset_path() -> Path:
    """JPEG court diagram (public/badminton-court-vector.jpeg)."""
    root = Path(__file__).resolve().parents[3]
    for candidate in (
        root / "public" / "badminton-court-vector.jpeg",
        root / "badminton-court-vector.jpeg",
        root / "public" / "badminton-court-vector.avif",
        root / "badminton-court-vector.avif",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Badminton court asset not found. Add public/badminton-court-vector.jpeg."
    )


def render_badminton_court_template(
    *,
    length_m: float = 13.4,
    width_m: float = 6.1,
    pixels_per_meter: int = DEFAULT_PIXELS_PER_METER,
) -> np.ndarray:
    """Load badminton-court-vector.jpeg, resized for heatmap / template API."""
    target_w = int(round(length_m * pixels_per_meter))
    target_h = int(round(width_m * pixels_per_meter))
    if target_w < 1 or target_h < 1:
        raise ValueError("Invalid badminton court template dimensions.")

    asset = badminton_court_asset_path()
    raw = np.fromfile(asset, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode badminton court image: {asset}")
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def is_badminton_court(length_m: float, width_m: float) -> bool:
    """True when dimensions match configured badminton court size."""
    from backend.app.pipeline.sports.config import calibration_matches_sport

    return calibration_matches_sport(
        sport="badminton",
        pitch_length_m=length_m,
        pitch_width_m=width_m,
    )


def render_court_template(
    *,
    length_m: float,
    width_m: float,
    pixels_per_meter: int = DEFAULT_PIXELS_PER_METER,
) -> np.ndarray:
    """Top-down court diagram for heatmap / template API (sport-aware)."""
    if is_badminton_court(length_m, width_m):
        return render_badminton_court_template(
            length_m=length_m,
            width_m=width_m,
            pixels_per_meter=pixels_per_meter,
        )
    return render_pitch_template(
        length_m=length_m,
        width_m=width_m,
        pixels_per_meter=pixels_per_meter,
    )


def warp_frame_to_topview(
    frame: np.ndarray,
    cal: PitchCalibration,
    *,
    pixels_per_meter: int = DEFAULT_PIXELS_PER_METER,
) -> np.ndarray:
    """Warp a broadcast frame into a top-down pitch-aligned view (verification)."""
    out_w = int(round(cal.pitch_length_m * pixels_per_meter))
    out_h = int(round(cal.pitch_width_m * pixels_per_meter))
    ppm = float(pixels_per_meter)

    dst_corners = (world_corners_meters(cal.pitch_length_m, cal.pitch_width_m) * ppm).astype(
        np.float32
    )
    src_corners = np.float32(cal.image_corners)
    H_frame_to_top = cv2.getPerspectiveTransform(src_corners, dst_corners)
    return cv2.warpPerspective(frame, H_frame_to_top, (out_w, out_h))


def draw_corners_overlay(
    frame: np.ndarray,
    corners: list[list[float]],
    *,
    boundary_points: list[list[float]] | None = None,
) -> np.ndarray:
    out = frame.copy()
    if boundary_points:
        bpts = np.array(boundary_points, dtype=np.int32)
        for i, (x, y) in enumerate(bpts):
            cv2.circle(out, (int(x), int(y)), 6, (255, 200, 0), -1)
            cv2.putText(
                out,
                str(i + 1),
                (int(x) + 8, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 200, 0),
                1,
                cv2.LINE_AA,
            )
        cv2.polylines(out, [bpts], True, (255, 200, 0), 2, cv2.LINE_AA)
    if not corners:
        return out
    pts = np.array(corners, dtype=np.int32)
    colors = [(0, 255, 255), (0, 165, 255), (0, 0, 255), (255, 0, 255)]
    for i in range(4):
        color = colors[i]
        x, y = int(pts[i, 0]), int(pts[i, 1])
        cv2.circle(out, (x, y), 8, color, -1)
        cv2.putText(
            out,
            CORNER_LABELS[i],
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.polylines(out, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
    return out
