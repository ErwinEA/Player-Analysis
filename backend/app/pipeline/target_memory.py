"""Persistent target fingerprint for lock persistence and re-acquisition."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.app.pipeline.segmentation import (
    _bbox_center,
    _center_dist_norm,
    is_valid_reid_candidate_bbox,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from backend.app.pipeline.pitch_homography import PitchCalibration

logger = logging.getLogger(__name__)

MemoryState = str  # UNLOCKED | LOCK_CANDIDATE | LOCKED | SEARCHING | REACQUIRED


def target_memory_enabled() -> bool:
    return os.environ.get("TARGET_MEMORY_ENABLED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def memory_debug_enabled() -> bool:
    return os.environ.get("TARGET_MEMORY_DEBUG", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def memory_jersey_min_conf() -> float:
    return float(os.environ.get("MEMORY_JERSEY_MIN_CONF", "0.35"))


def memory_score_thresh() -> float:
    return float(os.environ.get("MEMORY_SCORE_THRESH", "0.55"))


def memory_confidence_lock_thresh() -> float:
    """Rolling fingerprint confidence to promote LOCK_CANDIDATE → matcher lock."""
    return float(os.environ.get("MEMORY_CONF_LOCK_THRESH", "0.55"))


def memory_searching_frames() -> int:
    return max(1, int(os.environ.get("MEMORY_SEARCHING_FRAMES", "3")))


def _weight(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


@dataclass
class TargetMemory:
    jersey_number: int
    appearance_embedding: np.ndarray | None = None
    team_color_primary: str = ""
    team_color_secondary: str = ""
    avg_bbox_size: float = 0.0
    last_bbox: list[float] | None = None
    last_center: tuple[float, float] | None = None
    last_seen_frame: int = -1
    confidence: float = 0.0
    state: MemoryState = "UNLOCKED"
    locked_track_id: int | None = None
    miss_streak: int = 0
    reacquire_count: int = 0
    _size_samples: int = 0

    def seed_from_jersey(
        self,
        *,
        track_id: int,
        bbox: list[float],
        frame_idx: int,
        embedding: np.ndarray | None,
        jersey_conf: float,
    ) -> None:
        if jersey_conf < memory_jersey_min_conf():
            return
        self._set_embedding(embedding)
        self._update_geometry(bbox, frame_idx)
        self.locked_track_id = track_id
        self.confidence = max(self.confidence, jersey_conf)
        if self.state == "UNLOCKED":
            self.state = "LOCK_CANDIDATE"
        if memory_debug_enabled():
            logger.info(
                "TargetMemory seed track=%s frame=%s jersey_conf=%.2f state=%s",
                track_id,
                frame_idx,
                jersey_conf,
                self.state,
            )

    def on_lock(self, track_id: int, frame_idx: int, *, method: str) -> None:
        self.locked_track_id = track_id
        self.state = "LOCKED"
        self.miss_streak = 0
        self.confidence = max(self.confidence, 0.6)
        if memory_debug_enabled():
            logger.info(
                "TargetMemory lock formed track=%s frame=%s reason=%s conf=%.2f",
                track_id,
                frame_idx,
                method,
                self.confidence,
            )

    def on_visible(
        self,
        track_id: int,
        bbox: list[float],
        frame_idx: int,
        *,
        embedding: np.ndarray | None = None,
        jersey_conf: float = 0.0,
        color_score: float = 0.0,
    ) -> None:
        self._update_geometry(bbox, frame_idx)
        if embedding is not None:
            self._blend_embedding(embedding)
        if jersey_conf >= memory_jersey_min_conf():
            self.confidence = min(1.0, self.confidence + jersey_conf * 0.4)
        elif self.state in ("LOCKED", "SEARCHING", "REACQUIRED"):
            self.confidence = max(0.0, self.confidence - 0.05)
        if color_score > 0:
            self.confidence = min(1.0, self.confidence + color_score * 0.1)
        if track_id != self.locked_track_id and self.state == "SEARCHING":
            self.state = "REACQUIRED"
            self.reacquire_count += 1
            if memory_debug_enabled():
                logger.info(
                    "TargetMemory reacquired old_track=%s new_track=%s frame=%s",
                    self.locked_track_id,
                    track_id,
                    frame_idx,
                )
        self.locked_track_id = track_id
        self.miss_streak = 0
        if self.state in ("LOCK_CANDIDATE", "REACQUIRED"):
            self.state = "LOCKED"

    def on_miss(self, frame_idx: int) -> None:
        if self.state not in ("LOCKED", "REACQUIRED", "SEARCHING"):
            return
        self.miss_streak += 1
        self.confidence = max(0.0, self.confidence - 0.05)
        if self.miss_streak >= memory_searching_frames():
            self.state = "SEARCHING"
            if memory_debug_enabled():
                logger.info(
                    "TargetMemory searching frame=%s miss_streak=%s conf=%.2f",
                    frame_idx,
                    self.miss_streak,
                    self.confidence,
                )

    def reset(self) -> None:
        self.state = "UNLOCKED"
        self.locked_track_id = None
        self.miss_streak = 0
        self.confidence = 0.0
        self.last_bbox = None
        self.last_center = None
        self.last_seen_frame = -1

    def _set_embedding(self, embedding: np.ndarray | None) -> None:
        if embedding is None:
            return
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm <= 0:
            return
        self.appearance_embedding = vec / norm

    def _blend_embedding(self, embedding: np.ndarray) -> None:
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm <= 0:
            return
        vec = vec / norm
        if self.appearance_embedding is None:
            self.appearance_embedding = vec
            return
        blended = 0.8 * self.appearance_embedding + 0.2 * vec
        bn = float(np.linalg.norm(blended))
        if bn > 0:
            self.appearance_embedding = blended / bn

    def _update_geometry(self, bbox: list[float], frame_idx: int) -> None:
        x1, y1, x2, y2 = bbox
        area = max(0.0, (x2 - x1) * (y2 - y1))
        self._size_samples += 1
        alpha = 0.2
        if self._size_samples == 1:
            self.avg_bbox_size = area
        else:
            self.avg_bbox_size = (1 - alpha) * self.avg_bbox_size + alpha * area
        self.last_bbox = list(bbox)
        self.last_center = _bbox_center(bbox)
        self.last_seen_frame = frame_idx

    def resolve_candidate(
        self,
        tracks: list[dict[str, Any]],
        frame: NDArray,
        frame_idx: int,
        *,
        reid_extractor: object | None,
        calibration: PitchCalibration | None = None,
        ocr: object | None = None,
        frame_width: int = 0,
        frame_height: int = 0,
        get_color_score: Any | None = None,
    ) -> tuple[int, list[float], bool, dict[str, float]] | None:
        """Score visible tracks against fingerprint; return best above threshold."""
        if self.appearance_embedding is None and self.last_bbox is None:
            return None
        if not tracks:
            return None

        from backend.app.pipeline.reid import cosine_similarity
        from backend.app.pipeline.ocr import JerseyOCR

        w_app = _weight("MEMORY_W_APPEARANCE", 0.40)
        w_pos = _weight("MEMORY_W_POSITION", 0.30)
        w_color = _weight("MEMORY_W_COLOR", 0.20)
        w_jersey = _weight("MEMORY_W_JERSEY", 0.10)
        frame_shape = frame.shape[:2]
        ref_bbox = self.last_bbox
        best: tuple[float, int, list[float], dict[str, float]] | None = None

        for tr in tracks:
            tid = int(tr["track_id"])
            bbox = list(tr["bbox"])
            if not is_valid_reid_candidate_bbox(
                bbox, frame_shape, calibration=calibration
            ):
                continue

            appearance = 0.0
            if self.appearance_embedding is not None and reid_extractor is not None:
                emb = reid_extractor.embed(JerseyOCR.full_kit_crop(frame, bbox))
                appearance = cosine_similarity(emb, self.appearance_embedding)

            position = 0.0
            if ref_bbox is not None and frame_width > 0 and frame_height > 0:
                dist = _center_dist_norm(bbox, ref_bbox, frame_width, frame_height)
                position = max(0.0, 1.0 - dist / max(reid_fallback_max_center_frac(), 0.01))

            color = 0.0
            if get_color_score is not None:
                color = float(get_color_score(frame, bbox))

            jersey = 0.0
            if (
                ocr is not None
                and self.jersey_number > 0
            ):
                num, conf, _ = ocr.read_number_detailed(
                    frame, bbox, target_number=self.jersey_number
                )
                if num == self.jersey_number and conf >= memory_jersey_min_conf():
                    jersey = conf

            total = (
                appearance * w_app
                + position * w_pos
                + color * w_color
                + jersey * w_jersey
            )
            detail = {
                "appearance": round(appearance, 3),
                "position": round(position, 3),
                "color": round(color, 3),
                "jersey": round(jersey, 3),
                "total": round(total, 3),
            }
            if memory_debug_enabled() and total >= memory_score_thresh() * 0.8:
                logger.info(
                    "TargetMemory candidate track=%s frame=%s "
                    "appearance=%.2f position=%.2f color=%.2f jersey=%.2f total=%.2f",
                    tid,
                    frame_idx,
                    detail["appearance"],
                    detail["position"],
                    detail["color"],
                    detail["jersey"],
                    detail["total"],
                )
            if best is None or total > best[0]:
                best = (total, tid, bbox, detail)

        if best is None or best[0] < memory_score_thresh():
            return None
        _score, tid, bbox, detail = best
        is_fallback = tid != self.locked_track_id
        return tid, bbox, is_fallback, detail


def reid_fallback_max_center_frac() -> float:
    from backend.app.pipeline.segmentation import reid_fallback_max_center_frac as _frac

    return _frac()
