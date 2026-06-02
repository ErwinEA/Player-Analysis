from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


def clip_bbox(bbox: list[float], frame_shape: tuple[int, ...]) -> list[int]:
    """Clip a bbox [x1, y1, x2, y2] to frame bounds and return integer coords."""
    if len(frame_shape) < 2:
        return [0, 0, 0, 0]
    fh, fw = int(frame_shape[0]), int(frame_shape[1])
    x1 = max(0, min(fw - 1, int(bbox[0])))
    y1 = max(0, min(fh - 1, int(bbox[1])))
    x2 = max(0, min(fw, int(bbox[2])))
    y2 = max(0, min(fh, int(bbox[3])))
    if x2 <= x1:
        x2 = min(fw, x1 + 1)
    if y2 <= y1:
        y2 = min(fh, y1 + 1)
    return [x1, y1, x2, y2]


def _bbox_area(bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def is_valid_person_bbox(
    bbox: list[float],
    frame_shape: tuple[int, ...],
    *,
    min_size: int | None = None,
    max_height_frac: float | None = None,
    max_width_frac: float | None = None,
    max_area_frac: float | None = None,
    min_clipped_area_ratio: float | None = None,
    min_aspect: float | None = None,
) -> bool:
    """Reject degenerate ByteTrack/YOLO boxes (off-screen giants, slivers)."""
    if len(frame_shape) < 2:
        return False
    fh, fw = int(frame_shape[0]), int(frame_shape[1])
    min_size = min_size if min_size is not None else int(os.environ.get("BBOX_MIN_SIZE", "48"))
    max_height_frac = max_height_frac if max_height_frac is not None else float(
        os.environ.get("BBOX_MAX_HEIGHT_FRAC", "0.85")
    )
    max_width_frac = max_width_frac if max_width_frac is not None else float(
        os.environ.get("BBOX_MAX_WIDTH_FRAC", "0.75")
    )
    max_area_frac = max_area_frac if max_area_frac is not None else float(
        os.environ.get("BBOX_MAX_AREA_FRAC", "0.55")
    )
    min_clipped_area_ratio = min_clipped_area_ratio if min_clipped_area_ratio is not None else float(
        os.environ.get("BBOX_MIN_CLIPPED_AREA_RATIO", "0.55")
    )
    min_aspect = min_aspect if min_aspect is not None else float(os.environ.get("BBOX_MIN_ASPECT", "0.85"))

    raw_area = _bbox_area(bbox)
    if raw_area <= 0:
        return False

    clipped = clip_bbox(bbox, frame_shape)
    w = clipped[2] - clipped[0]
    h = clipped[3] - clipped[1]
    if w < min_size or h < min_size:
        return False

    clipped_area = float(w * h)
    if clipped_area / raw_area < min_clipped_area_ratio:
        return False

    if h > fh * max_height_frac or w > fw * max_width_frac:
        return False

    frame_area = float(fh * fw)
    if clipped_area / frame_area > max_area_frac:
        return False

    aspect = h / max(w, 1)
    if aspect < min_aspect:
        return False

    return True


def normalize_bbox(bbox: list[float], frame_shape: tuple[int, ...]) -> list[float]:
    """Clip bbox to frame and return float coords for storage/scoring."""
    c = clip_bbox(bbox, frame_shape)
    return [float(c[0]), float(c[1]), float(c[2]), float(c[3])]


def crop_region(
    frame: NDArray,
    bbox: list[float],
    *,
    top_pct: float,
    bot_pct: float,
    pad_x_pct: float = 0.0,
) -> NDArray:
    """Slice a horizontal band from a person bbox."""
    if frame.size == 0:
        return frame
    clipped = clip_bbox(bbox, frame.shape)
    x1, y1, x2, y2 = clipped
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    pad_x = int(pad_x_pct * w)
    y_top = y1 + int(top_pct * h)
    y_bot = y1 + int(bot_pct * h)
    if y_bot <= y_top:
        y_bot = y_top + 1
    x_left = max(0, x1 - pad_x)
    x_right = min(frame.shape[1], x2 + pad_x)
    return frame[y_top:y_bot, x_left:x_right]
