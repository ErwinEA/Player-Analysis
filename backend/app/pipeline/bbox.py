from __future__ import annotations

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


def is_valid_bbox(bbox: list[float], frame_shape: tuple[int, ...], min_size: int = 8) -> bool:
    """Return True if bbox has positive area inside frame and at least min_size on each side."""
    if len(frame_shape) < 2:
        return False
    clipped = clip_bbox(bbox, frame_shape)
    w = clipped[2] - clipped[0]
    h = clipped[3] - clipped[1]
    return w >= min_size and h >= min_size


def crop_region(
    frame: NDArray,
    bbox: list[float],
    *,
    top_pct: float,
    bot_pct: float,
    pad_x_pct: float = 0.0,
) -> NDArray:
    """
    Slice a horizontal band from a person bbox.

    top_pct / bot_pct are fractions of the bbox height (0.0 = bbox top, 1.0 = bbox bottom).
    pad_x_pct widens the slice horizontally by that fraction of bbox width on each side.
    """
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
