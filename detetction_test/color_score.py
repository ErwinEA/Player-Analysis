from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from bbox_utils import crop_region

if TYPE_CHECKING:
    from numpy.typing import NDArray

_HIST_BINS = (8, 8)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_color!r}")
    try:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError as exc:
        raise ValueError(f"Invalid hex color: {hex_color!r}") from exc


def hex_pixel_match_score(
    crop: NDArray, hex_color: str, tolerance: int = 10
) -> float:
    """Fraction of pixels within +/- tolerance of target hex (BGR crop)."""
    if crop.size == 0:
        return 0.0
    r, g, b = _hex_to_rgb(hex_color)
    target = np.array([b, g, r], dtype=np.int16)
    pixels = crop.reshape(-1, crop.shape[-1]).astype(np.int16)
    diff = np.abs(pixels - target)
    matches = np.all(diff <= tolerance, axis=1)
    if matches.size == 0:
        return 0.0
    return float(matches.mean())


def jersey_color_score(
    frame: NDArray,
    bbox: list[float],
    primary_hex: str,
    secondary_hex: str,
    tolerance: int = 10,
) -> float:
    """Best primary/secondary pixel-match score over torso+shorts region."""
    crop = crop_region(frame, bbox, top_pct=0.10, bot_pct=0.90, pad_x_pct=0.05)
    if crop.size == 0:
        return 0.0
    primary_score = hex_pixel_match_score(crop, primary_hex, tolerance)
    secondary_score = hex_pixel_match_score(crop, secondary_hex, tolerance)
    return max(primary_score, secondary_score)


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    r, g, b = _hex_to_rgb(hex_color)
    return b, g, r


def _hsv_histogram(crop_bgr: NDArray) -> np.ndarray:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [_HIST_BINS[0], _HIST_BINS[1]],
        [0, 180, 0, 256],
    )
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def _reference_kit_histogram(primary_hex: str, secondary_hex: str, size: int = 64) -> np.ndarray:
    """Synthetic two-band swatch histogram for the target kit profile."""
    swatch = np.zeros((size, size, 3), dtype=np.uint8)
    swatch[: size // 2, :] = _hex_to_bgr(primary_hex)
    swatch[size // 2 :, :] = _hex_to_bgr(secondary_hex)
    return _hsv_histogram(swatch)


def kit_torso_primary_match(
    frame: NDArray,
    bbox: list[float],
    primary_hex: str,
    tolerance: int = 40,
) -> float:
    """Primary jersey color pixel fraction on upper-torso crop (excludes shorts/grass)."""
    crop = crop_region(frame, bbox, top_pct=0.10, bot_pct=0.55, pad_x_pct=0.05)
    if crop.size == 0:
        return 0.0
    return hex_pixel_match_score(crop, primary_hex, tolerance)


def kit_histogram_match_score(
    frame: NDArray,
    bbox: list[float],
    primary_hex: str,
    secondary_hex: str,
) -> float:
    """Histogram correlation between torso crop and target kit profile (0–1)."""
    crop = crop_region(frame, bbox, top_pct=0.10, bot_pct=0.90, pad_x_pct=0.05)
    if crop.size == 0:
        return 0.0
    crop_hist = _hsv_histogram(crop)
    ref_hist = _reference_kit_histogram(primary_hex, secondary_hex)
    score = cv2.compareHist(crop_hist, ref_hist, cv2.HISTCMP_CORREL)
    return max(0.0, float(score))
