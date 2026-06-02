"""Resolve target foot + identification tier for ball-event sampling."""

from __future__ import annotations

import os
from dataclasses import dataclass

from backend.app.pipeline.bbox import is_valid_bbox
from backend.app.pipeline.movement_stats import foot_position_pixels
from backend.app.pipeline.pitch_homography import PitchCalibration
from backend.app.pipeline.segmentation import seg_jersey_ocr_min_conf


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class BallTargetObservation:
    foot_px: tuple[float, float]
    foot_m: tuple[float, float] | None
    target_visible: bool
    segment_source: str
    identification_type: str  # locked | ocr | reid | track
    track_id: int


def _foot_from_bbox(
    bbox: list[float],
    calibration: PitchCalibration | None,
) -> tuple[tuple[float, float], tuple[float, float] | None]:
    fx, fy = foot_position_pixels(bbox)
    foot_px = (fx, fy)
    foot_m = None
    if calibration is not None:
        try:
            foot_m = calibration.pixel_to_meters(fx, fy)
        except ValueError:
            foot_m = None
    return foot_px, foot_m


def _track_bbox(tracks: list[dict], track_id: int) -> list[float] | None:
    for tr in tracks:
        if int(tr["track_id"]) == track_id:
            return list(tr["bbox"])
    return None


def resolve_ball_event_target(
    *,
    tracks: list[dict],
    matcher,
    lock,
    calibration: PitchCalibration | None,
    frame_shape: tuple[int, ...],
    frame_ocr_match: tuple[int, float] | None,
    carry_track_id: int | None,
) -> tuple[BallTargetObservation | None, int | None]:
    """Foot + id tier for ball events. Returns (observation, updated_carry_track_id)."""
    if lock is not None:
        tid = lock.track_id
        bbox = _track_bbox(tracks, tid)
        if bbox is None or not is_valid_bbox(bbox, frame_shape, min_size=16):
            return None, carry_track_id
        foot_px, foot_m = _foot_from_bbox(bbox, calibration)
        return (
            BallTargetObservation(
                foot_px=foot_px,
                foot_m=foot_m,
                target_visible=True,
                segment_source="bbox_track",
                identification_type="locked",
                track_id=tid,
            ),
            tid,
        )

    min_ocr = seg_jersey_ocr_min_conf()
    if frame_ocr_match is not None:
        tid, conf = frame_ocr_match
        bbox = _track_bbox(tracks, tid)
        if bbox is not None and is_valid_bbox(bbox, frame_shape, min_size=16):
            foot_px, foot_m = _foot_from_bbox(bbox, calibration)
            return (
                BallTargetObservation(
                    foot_px=foot_px,
                    foot_m=foot_m,
                    target_visible=True,
                    segment_source="bbox_track",
                    identification_type="ocr",
                    track_id=tid,
                ),
                tid,
            )

    soft = matcher.soft_identify_track(
        prefer_track_id=carry_track_id,
        reid_threshold=_env_float("SOFT_REID_THRESH", 0.5),
    )
    if soft is None:
        return None, carry_track_id

    tid, id_type, _score = soft
    bbox = _track_bbox(tracks, tid)
    if bbox is None or not is_valid_bbox(bbox, frame_shape, min_size=16):
        return None, carry_track_id
    foot_px, foot_m = _foot_from_bbox(bbox, calibration)
    return (
        BallTargetObservation(
            foot_px=foot_px,
            foot_m=foot_m,
            target_visible=True,
            segment_source="bbox_track",
            identification_type=id_type,
            track_id=tid,
        ),
        tid,
    )
