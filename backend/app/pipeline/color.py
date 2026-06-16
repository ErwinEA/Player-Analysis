from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

from backend.app.pipeline.bbox import crop_region

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _default_tolerance() -> int:
    return int(os.environ.get("COLOR_TOLERANCE", "10"))


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#").strip()
    if not h:
        raise ValueError("empty hex color")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _use_hsv_color(use_hsv: bool | None = None) -> bool:
    if use_hsv is not None:
        return use_hsv
    return os.environ.get("COLOR_USE_HSV", "0").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def hex_pixel_match_score(
    crop: NDArray, hex_color: str, tolerance: int | None = None
) -> float:
    """
    Fraction of pixels in `crop` whose R, G, B channels are all within +/- tolerance
    units of the target hex color.

    crop is BGR (OpenCV convention). Returns 0.0 .. 1.0.
    """
    if crop.size == 0:
        return 0.0
    tol = _default_tolerance() if tolerance is None else int(tolerance)
    r, g, b = _hex_to_rgb(hex_color)
    target = np.array([b, g, r], dtype=np.int16)  # BGR order
    pixels = crop.reshape(-1, crop.shape[-1]).astype(np.int16)
    diff = np.abs(pixels - target)
    matches = np.all(diff <= tol, axis=1)
    if matches.size == 0:
        return 0.0
    return float(matches.mean())


def hex_pixel_match_score_hsv(
    crop: NDArray, hex_color: str, tolerance: int | None = None
) -> float:
    """HSV hue-channel match (more robust to lighting than RGB tolerance)."""
    if crop.size == 0:
        return 0.0
    import cv2

    tol = _default_tolerance() if tolerance is None else int(tolerance)
    r, g, b = _hex_to_rgb(hex_color)
    target_bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0, 0]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    hue_diff = np.abs(pixels[:, 0].astype(np.int16) - int(target_hsv[0]))
    hue_diff = np.minimum(hue_diff, 180 - hue_diff)
    sat_ok = np.abs(pixels[:, 1].astype(np.int16) - int(target_hsv[1])) <= tol * 2
    val_ok = np.abs(pixels[:, 2].astype(np.int16) - int(target_hsv[2])) <= tol * 2
    matches = (hue_diff <= max(tol, 8)) & sat_ok & val_ok
    if matches.size == 0:
        return 0.0
    return float(matches.mean())


def jersey_color_score(
    frame: NDArray,
    bbox: list[float],
    primary_hex: str,
    secondary_hex: str,
    tolerance: int | None = None,
    *,
    use_hsv: bool | None = None,
) -> float:
    """
    Best of (primary, secondary) pixel-match scores over the player's torso+shorts region.
    Higher means more of the player's kit pixels lie within tolerance of one of the colors.
    """
    crop = crop_region(frame, bbox, top_pct=0.10, bot_pct=0.90, pad_x_pct=0.05)
    if crop.size == 0:
        return 0.0
    score_fn = (
        hex_pixel_match_score_hsv
        if _use_hsv_color(use_hsv)
        else hex_pixel_match_score
    )
    primary_score = score_fn(crop, primary_hex, tolerance)
    if not secondary_hex or not secondary_hex.strip():
        return primary_score
    secondary_score = score_fn(crop, secondary_hex, tolerance)
    return max(primary_score, secondary_score)
