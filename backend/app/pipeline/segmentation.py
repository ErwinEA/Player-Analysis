"""MobileSAM segmentation for locked target player (legacy pipeline)."""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from backend.app.pipeline.bbox import is_valid_person_bbox
from backend.app.pipeline.mask_rle import MaskRle, encode_mask_rle
from backend.app.pipeline.movement_stats import foot_position_pixels
from backend.app.pipeline.sam_weights import resolve_sam_weights

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from backend.app.pipeline.pitch_homography import PitchCalibration

logger = logging.getLogger(__name__)

SegmentSource = Literal["sam", "sam_retry", "bbox_fallback", "reid_fallback"]

_segmenter: MobileSamSegmenter | None = None
_segmenter_unavailable = False
_segmenter_unavailable_reason: str | None = None
_segmenter_load_attempted_at: float = 0.0
_SEGMENTER_RETRY_SECONDS: float = 30.0

_MIN_MASK_AREA_RATIO = 0.002
_BOTTOM_BAND_FRAC = 0.05


def sam_enabled() -> bool:
    return os.environ.get("SAM_ENABLED", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def sam_warmup_frames() -> int:
    return max(1, int(os.environ.get("SAM_WARMUP_FRAMES", "1")))


def reid_fallback_thresh() -> float:
    return float(os.environ.get("REID_FALLBACK_THRESH", "0.65"))


def reid_fallback_margin() -> float:
    return float(os.environ.get("REID_FALLBACK_MARGIN", "0.10"))


def reid_fallback_max_center_frac() -> float:
    """Max normalized center jump vs last good bbox for ReID fallback."""
    return float(os.environ.get("REID_FALLBACK_MAX_CENTER_FRAC", "0.12"))


def sam_mask_scale() -> float:
    """Downscale factor for API mask RLE (1.0 = full resolution)."""
    raw = os.environ.get("SAM_MASK_SCALE", "0.5").strip()
    try:
        scale = float(raw)
    except ValueError:
        scale = 0.5
    return min(1.0, max(0.125, scale))


def downscale_mask_for_api(mask: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return mask
    import cv2

    h, w = mask.shape
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    small = cv2.resize(
        mask.astype(np.uint8),
        (nw, nh),
        interpolation=cv2.INTER_NEAREST,
    )
    return small.astype(bool)


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _center_dist_norm(
    a: list[float],
    b: list[float],
    frame_w: int,
    frame_h: int,
) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    diag = (frame_w**2 + frame_h**2) ** 0.5
    if diag <= 0:
        return 0.0
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 / diag


def _resolve_device() -> str:
    raw = os.environ.get("SAM_DEVICE", "auto").strip().lower()
    if raw in ("cuda", "mps", "cpu"):
        return raw
    try:
        import torch

        if raw == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        return "cpu"
    except Exception:
        return "cpu"


def foot_from_mask(mask: np.ndarray) -> tuple[float, float] | None:
    """Foot at lowest foreground row; x = median x of bottom 5% of mask pixels."""
    if mask.ndim != 2 or not np.any(mask):
        return None
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y_max = int(ys.max())
    y_min_band = y_max - max(1, int(np.ceil(mask.shape[0] * _BOTTOM_BAND_FRAC)))
    in_band = ys >= y_min_band
    if not np.any(in_band):
        in_band = ys == y_max
    foot_x = float(np.median(xs[in_band]))
    foot_y = float(y_max)
    return foot_x, foot_y


def mask_bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(xs.min()),
        float(ys.min()),
        float(xs.max()),
        float(ys.max()),
    ]


def crop_from_mask(frame: NDArray, mask: np.ndarray) -> NDArray:
    """Tight crop around mask foreground."""
    x1, y1, x2, y2 = mask_bbox(mask)
    h, w = frame.shape[:2]
    x1i = max(0, int(x1))
    y1i = max(0, int(y1))
    x2i = min(w, int(np.ceil(x2)))
    y2i = min(h, int(np.ceil(y2)))
    if x2i <= x1i or y2i <= y1i:
        return frame[0:0, 0:0]
    return frame[y1i:y2i, x1i:x2i].copy()


@dataclass
class SegmentResult:
    mask: np.ndarray | None
    mask_rle: MaskRle | None
    foot_px: tuple[float, float]
    foot_nx: float
    foot_ny: float
    segment_source: SegmentSource
    mask_fallback: bool
    segment_track_id: int


@dataclass
class SegmentationState:
    """Warmup and SAM-active flags after identity lock."""

    lock_track_id: int | None = None
    visible_count: int = 0
    sam_active: bool = False
    segmentation_started_at_frame: int | None = None
    warned_unavailable: bool = False

    def acquire_lock(self, track_id: int) -> None:
        """Reset warmup when the locked track id changes (not after SAM is active)."""
        if self.sam_active:
            self.lock_track_id = track_id
            return
        if self.lock_track_id != track_id:
            self.lock_track_id = track_id
            self.visible_count = 0
            self.sam_active = False
            self.segmentation_started_at_frame = None

    def record_visible_frame(self, frame_idx: int) -> None:
        """Count a frame where the locked track is visible; activates SAM at threshold."""
        if self.sam_active:
            return
        self.visible_count += 1
        if self.visible_count >= sam_warmup_frames():
            self.sam_active = True
            self.segmentation_started_at_frame = frame_idx

    def reset_visible_if_missing(self) -> None:
        if not self.sam_active:
            self.visible_count = 0


class MobileSamSegmenter:
    """Lazy MobileSAM predictor (box + center-point prompts)."""

    def __init__(self, weights_path: str, device: str) -> None:
        import cv2
        import torch

        self._cv2 = cv2
        self._torch = torch
        self.device = device

        try:
            from mobile_sam import sam_model_registry, SamPredictor
        except ImportError:
            repo = os.environ.get("MOBILE_SAM_REPO", "").strip()
            if repo and repo not in sys.path:
                sys.path.insert(0, repo)
            from mobile_sam import sam_model_registry, SamPredictor

        model_type = "vit_t"
        sam = sam_model_registry[model_type](checkpoint=weights_path)
        sam.to(device=device)
        sam.eval()
        self._predictor = SamPredictor(sam)

    def _set_image(self, frame_bgr: NDArray) -> None:
        rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        self._predictor.set_image(rgb)

    def _mask_from_predict(
        self,
        frame_shape: tuple[int, ...],
        *,
        box: list[float] | None = None,
        center: tuple[float, float] | None = None,
    ) -> np.ndarray | None:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        if box is not None:
            x1, y1, x2, y2 = box
            input_box = np.array([x1, y1, x2, y2], dtype=np.float32)
            masks, _, _ = self._predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
        elif center is not None:
            cx, cy = center
            point_coords = np.array([[cx, cy]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)
            masks, _, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=None,
                multimask_output=False,
            )
        else:
            return None

        if masks is None or len(masks) == 0:
            return None
        mask = masks[0]
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = self._cv2.resize(
                mask.astype(np.uint8),
                (w, h),
                interpolation=self._cv2.INTER_NEAREST,
            ).astype(bool)
        area = float(mask.sum())
        if area < _MIN_MASK_AREA_RATIO * h * w:
            return None
        return mask.astype(bool)

    def segment(
        self,
        frame: NDArray,
        bbox: list[float],
    ) -> tuple[np.ndarray | None, SegmentSource]:
        self._set_image(frame)
        mask = self._mask_from_predict(frame.shape, box=bbox)
        if mask is not None:
            return mask, "sam"
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        mask = self._mask_from_predict(frame.shape, center=(cx, cy))
        if mask is not None:
            return mask, "sam_retry"
        return None, "bbox_fallback"


def prepare_mobile_sam_for_run() -> None:
    """Allow MobileSAM to load again on the next get_mobile_sam_segmenter() call.

    The API process caches a failed load in _segmenter_unavailable; resetting per
    analyze run lets installs/restarts take effect without restarting uvicorn.
    """
    global _segmenter_unavailable, _segmenter_unavailable_reason
    _segmenter_unavailable = False
    _segmenter_unavailable_reason = None


def mobile_sam_unavailable_reason() -> str | None:
    return _segmenter_unavailable_reason


def _mobile_sam_load_status() -> str:
    """ok | loading | error — for /health mobile_sam.status."""
    if not sam_enabled():
        return "ok"
    if _segmenter is not None:
        return "ok"
    if _segmenter_unavailable:
        return "error"
    return "loading"


def mobile_sam_status() -> dict[str, object]:
    """Diagnostics for /health and analyze responses (does not trigger load)."""
    weights = resolve_sam_weights()
    try:
        import mobile_sam  # noqa: F401

        import_ok = True
    except ImportError:
        import_ok = False
    status = _mobile_sam_load_status()
    return {
        "enabled": sam_enabled(),
        "import_ok": import_ok,
        "weights_found": weights is not None,
        "weights_path": str(weights) if weights is not None else None,
        "loaded": _segmenter is not None,
        "unavailable": _segmenter_unavailable,
        "unavailable_reason": _segmenter_unavailable_reason,
        "status": status,
    }


def mobile_sam_health() -> dict[str, object]:
    """Health payload; triggers lazy load via get_mobile_sam_segmenter()."""
    if sam_enabled():
        get_mobile_sam_segmenter()
    payload = mobile_sam_status()
    payload["status"] = _mobile_sam_load_status()
    return payload


def get_mobile_sam_segmenter() -> MobileSamSegmenter | None:
    global _segmenter, _segmenter_unavailable, _segmenter_unavailable_reason
    global _segmenter_load_attempted_at
    if not sam_enabled():
        return None
    if _segmenter is not None:
        return _segmenter
    if _segmenter_unavailable:
        elapsed = time.time() - _segmenter_load_attempted_at
        if elapsed < _SEGMENTER_RETRY_SECONDS:
            return None
        _segmenter_unavailable = False
        _segmenter_unavailable_reason = None

    _segmenter_load_attempted_at = time.time()
    weights = resolve_sam_weights()
    if weights is None:
        reason = (
            "MobileSAM weights not found (place mobile_sam.pt under backend/weights/ "
            "or set SAM_WEIGHTS)"
        )
        logger.warning("%s. Segmentation disabled.", reason)
        _segmenter_unavailable = True
        _segmenter_unavailable_reason = reason
        return None
    try:
        device = _resolve_device()
        _segmenter = MobileSamSegmenter(str(weights), device)
        logger.info("MobileSAM loaded from %s on %s", weights, device)
        return _segmenter
    except Exception as exc:
        reason = f"MobileSAM load failed: {exc}"
        logger.warning("%s", reason)
        _segmenter_unavailable = True
        _segmenter_unavailable_reason = reason
        return None


def is_valid_reid_candidate_bbox(
    bbox: list[float],
    frame_shape: tuple[int, ...],
    *,
    calibration: PitchCalibration | None = None,
) -> bool:
    """Reject ReID candidate boxes that cannot be the target player.

    Two gates: a person-shape gate (rejects scoreboard banners, grass slivers, giants)
    and, when calibration is available, an on-pitch gate on the projected foot point
    (rejects boxes whose feet land off the playing surface, e.g. crowd/graphics).
    """
    if not is_valid_person_bbox(bbox, frame_shape):
        return False
    if calibration is not None:
        from backend.app.pipeline.pitch_homography import is_on_pitch

        foot_x, foot_y = foot_position_pixels(bbox)
        try:
            x_m, y_m = calibration.pixel_to_meters(foot_x, foot_y)
        except ValueError:
            return False
        if not is_on_pitch(
            x_m,
            y_m,
            length_m=calibration.pitch_length_m,
            width_m=calibration.pitch_width_m,
        ):
            return False
    return True


def resolve_target_bbox(
    tracks: list[dict],
    lock_track_id: int,
    *,
    reid_prototype: np.ndarray | None,
    reid_extractor: object | None,
    frame: NDArray,
    last_bbox: list[float] | None = None,
    frame_width: int = 0,
    frame_height: int = 0,
    calibration: PitchCalibration | None = None,
) -> tuple[int, list[float], bool] | None:
    """Return (track_id, bbox, is_reid_fallback) for locked or ReID fallback track."""
    frame_shape = frame.shape[:2]
    by_id = {int(t["track_id"]): t for t in tracks}
    if lock_track_id in by_id:
        bbox = list(by_id[lock_track_id]["bbox"])
        if is_valid_reid_candidate_bbox(bbox, frame_shape, calibration=calibration):
            return lock_track_id, bbox, False
        # Locked track resolved onto an implausible region (graphic/off-pitch);
        # fall through to ReID rather than trust a bad box.

    if reid_prototype is None or reid_extractor is None:
        return None

    from backend.app.pipeline.reid import cosine_similarity
    from backend.app.pipeline.ocr import JerseyOCR

    scored: list[tuple[float, int, list[float]]] = []
    for tr in tracks:
        tid = int(tr["track_id"])
        bbox = list(tr["bbox"])
        if not is_valid_reid_candidate_bbox(bbox, frame_shape, calibration=calibration):
            continue
        kit = JerseyOCR.full_kit_crop(frame, bbox)
        emb = reid_extractor.embed(kit)
        sim = cosine_similarity(emb, reid_prototype)
        scored.append((sim, tid, bbox))

    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    best_sim, best_id, best_bbox = scored[0]
    thresh = reid_fallback_thresh()
    if best_sim < thresh:
        return None
    if len(scored) > 1 and best_sim - scored[1][0] < reid_fallback_margin():
        return None
    if last_bbox is not None and frame_width > 0 and frame_height > 0:
        if _center_dist_norm(best_bbox, last_bbox, frame_width, frame_height) > reid_fallback_max_center_frac():
            return None
    return best_id, best_bbox, True


def run_segmentation_for_target(
    frame: NDArray,
    bbox: list[float],
    track_id: int,
    *,
    is_reid_fallback: bool,
    width: int,
    height: int,
    segmenter: MobileSamSegmenter | None,
) -> SegmentResult:
    foot_px = foot_position_pixels(bbox)
    foot_nx = foot_px[0] / width if width > 0 else 0.0
    foot_ny = foot_px[1] / height if height > 0 else 0.0
    segment_source: SegmentSource = "reid_fallback" if is_reid_fallback else "sam"
    src: SegmentSource = segment_source
    mask_fallback = False

    mask: np.ndarray | None = None
    if segmenter is not None:
        mask, sam_src = segmenter.segment(frame, bbox)
        if mask is not None:
            if is_reid_fallback:
                src = "reid_fallback"
            else:
                src = sam_src
            foot = foot_from_mask(mask)
            if foot is not None:
                foot_px = foot
                foot_nx = foot_px[0] / width if width > 0 else 0.0
                foot_ny = foot_px[1] / height if height > 0 else 0.0
        else:
            mask_fallback = True
            src = "reid_fallback" if is_reid_fallback else "bbox_fallback"

    mask_rle = None
    if mask is not None:
        api_mask = downscale_mask_for_api(mask, sam_mask_scale())
        mask_rle = encode_mask_rle(api_mask)

    return SegmentResult(
        mask=mask,
        mask_rle=mask_rle,
        foot_px=foot_px,
        foot_nx=round(foot_nx, 4),
        foot_ny=round(foot_ny, 4),
        segment_source=src,
        mask_fallback=mask_fallback,
        segment_track_id=track_id,
    )
