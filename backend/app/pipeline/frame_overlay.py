"""cv2 track/ball overlays (foot ellipses, labels, locked highlight, ball marker)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from backend.app.pipeline.movement_stats import foot_position_pixels

# BGR palette (supervision-style cyan / yellow / white)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


@dataclass(frozen=True)
class TrackDraw:
    track_id: int
    bbox: list[float]
    confidence: float = 1.0
    label: str | None = None


def _clip_bbox(bbox: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def draw_foot_ellipse(
    img: np.ndarray,
    bbox: list[float],
    *,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
    if x2 <= x1 or y2 <= y1:
        return
    fx, fy = foot_position_pixels([float(x1), float(y1), float(x2), float(y2)])
    axis_x = max(4, int((x2 - x1) * 0.5))
    axis_y = max(3, int((x2 - x1) * 0.25))
    cv2.ellipse(
        img,
        (int(fx), int(fy)),
        (axis_x, axis_y),
        0,
        0,
        360,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_label_top_center(
    img: np.ndarray,
    bbox: list[float],
    text: str,
    *,
    color: tuple[int, int, int],
    text_color: tuple[int, int, int] = BLACK,
) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
    if x2 <= x1:
        return
    cx = (x1 + x2) // 2
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    (tw, th), _baseline = cv2.getTextSize(text, font, scale, thick)
    pad = 3
    left = cx - tw // 2 - pad
    top = max(0, y1 - th - pad * 2 - 4)
    cv2.rectangle(img, (left, top), (left + tw + pad * 2, top + th + pad * 2), color, -1)
    cv2.putText(
        img,
        text,
        (left + pad, top + th + pad - 1),
        font,
        scale,
        text_color,
        thick,
        cv2.LINE_AA,
    )


def draw_ball_triangle(
    img: np.ndarray,
    bbox: list[float],
    *,
    base: int = 20,
    height: int = 15,
    color: tuple[int, int, int] = WHITE,
) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
    if x2 <= x1:
        return
    cx = (x1 + x2) // 2
    apex = (cx, max(0, y1 - 2))
    left = (cx - base // 2, y1 + height)
    right = (cx + base // 2, y1 + height)
    pts = np.array([apex, left, right], dtype=np.int32)
    cv2.fillConvexPoly(img, pts, color, lineType=cv2.LINE_AA)


def draw_tracks(
    frame: np.ndarray,
    tracks: list[TrackDraw],
    *,
    locked_track_id: int | None,
    ball_bbox: list[float] | None = None,
) -> np.ndarray:
    """Return an annotated copy of frame (input is never mutated)."""
    out = frame.copy()

    for tr in tracks:
        if locked_track_id is not None and tr.track_id == locked_track_id:
            continue
        label = tr.label or str(tr.track_id)
        thickness = 1 if tr.confidence < 0.5 else 2
        draw_foot_ellipse(out, tr.bbox, color=CYAN, thickness=thickness)
        draw_label_top_center(out, tr.bbox, label, color=CYAN)

    if locked_track_id is not None:
        for tr in tracks:
            if tr.track_id != locked_track_id:
                continue
            label = tr.label or str(tr.track_id)
            draw_foot_ellipse(out, tr.bbox, color=YELLOW, thickness=3)
            draw_label_top_center(out, tr.bbox, label, color=YELLOW)

    if ball_bbox is not None:
        draw_ball_triangle(out, ball_bbox)

    return out


def draw_shuttle_dot(
    img: np.ndarray,
    cx: float,
    cy: float,
    *,
    radius: int = 5,
    color: tuple[int, int, int] = (0, 165, 255),
) -> None:
    h, w = img.shape[:2]
    ix, iy = int(round(cx)), int(round(cy))
    if 0 <= ix < w and 0 <= iy < h:
        cv2.circle(img, (ix, iy), radius, color, -1, lineType=cv2.LINE_AA)


def draw_rally_end_flash(
    img: np.ndarray,
    bbox: list[float],
    *,
    color: tuple[int, int, int] = WHITE,
    thickness: int = 3,
) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)


def draw_overlay_badminton(
    frame: np.ndarray,
    tracks: list[TrackDraw],
    *,
    locked_track_id: int | None,
    court_side_label: str,
    shuttle_px: tuple[float, float] | None = None,
    rally_state: str = "IDLE",
) -> np.ndarray:
    """Badminton annotated frame: court-side label, optional shuttle, rally tint."""
    out = frame.copy()
    locked_color = CYAN if rally_state == "LIVE" else YELLOW

    for tr in tracks:
        if locked_track_id is not None and tr.track_id == locked_track_id:
            continue
        thickness = 1 if tr.confidence < 0.5 else 2
        draw_foot_ellipse(out, tr.bbox, color=CYAN, thickness=thickness)
        draw_label_top_center(out, tr.bbox, str(tr.track_id), color=CYAN)

    if locked_track_id is not None:
        for tr in tracks:
            if tr.track_id != locked_track_id:
                continue
            label = court_side_label or tr.label or str(tr.track_id)
            draw_foot_ellipse(out, tr.bbox, color=locked_color, thickness=3)
            draw_label_top_center(out, tr.bbox, label, color=locked_color)
            if rally_state == "END":
                draw_rally_end_flash(out, tr.bbox)

    if shuttle_px is not None:
        draw_shuttle_dot(out, shuttle_px[0], shuttle_px[1])

    return out
