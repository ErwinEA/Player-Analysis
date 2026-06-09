from __future__ import annotations

import logging
import os
from collections import Counter, defaultdict
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

from backend.app.pipeline.bbox import is_valid_bbox
from backend.app.pipeline.color import jersey_color_score
from backend.app.pipeline.debug_ocr import OcrDebug
from backend.app.pipeline.detector import get_person_detector
from backend.app.pipeline.matcher import TrackIdentityMatcher
from backend.app.pipeline.ocr import JerseyOCR
from backend.app.pipeline.reid import get_reid_extractor
from backend.app.pipeline.scene_cut import SceneCutDetector
from backend.app.pipeline.tracker import PlayerTracker
from backend.app.pipeline.frame_overlay import TrackDraw, draw_tracks
from backend.app.pipeline.heatmap import build_heatmap_from_rows
from backend.app.pipeline.movement_stats import (
    compute_movement_stats_from_rows,
    foot_position_pixels,
)
from backend.app.pitch_api import resolve_calibration
from backend.app.video_store import ensure_browser_playable_mp4, open_browser_video_writer
from backend.app.pipeline.pitch_homography import PitchCalibration
from backend.app.pipeline.segmentation import (
    SegmentationState,
    get_mobile_sam_segmenter,
    mask_bbox,
    mobile_sam_unavailable_reason,
    prepare_mobile_sam_for_run,
    resolve_target_bbox,
    run_segmentation_for_target,
    sam_enabled,
    sam_warmup_frames,
    segment_result_from_bbox,
    should_run_sam_this_frame,
)
from backend.app.schemas import (
    AnalyzeResponse,
    HeatmapResult,
    MovementStats as MovementStatsSchema,
    PlayerDetails,
    InferredBallEvent,
    PlayerEventCounts,
    Row,
    TargetMatch,
    VideoMeta,
    jersey_colors_configured,
)


def _ocr_every() -> int:
    return max(1, int(os.environ.get("OCR_EVERY", "3")))


def _track_min_age() -> int:
    return max(1, int(os.environ.get("TRACK_MIN_AGE", "3")))


# Legacy pipeline cap: detection + tracking run on at most this many frames.
DEFAULT_MAX_FRAMES = 5000
MAX_FRAMES_ENV_CAP = 5000
UNLIMITED_FRAME_CAP = 10_000_000


def _max_frames() -> int:
    raw = os.environ.get("MAX_FRAMES", "").strip()
    if not raw:
        return DEFAULT_MAX_FRAMES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_FRAMES
    if value <= 0:
        return DEFAULT_MAX_FRAMES
    return min(max(1, value), MAX_FRAMES_ENV_CAP)


def _release_inference_memory() -> None:
    """Return GPU/MPS memory after a long run so the next analyze starts faster."""
    import gc

    gc.collect()
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _frame_start() -> int:
    """Skip leading frames (debug clips / late lock windows)."""
    raw = os.environ.get("FRAME_START", "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _include_masks_in_api_response() -> bool:
    """Whether /api/analyze should return mask_rle (overlay). Default on."""
    raw = os.environ.get("INCLUDE_MASKS_IN_API", "1").strip().lower()
    return raw not in ("0", "false", "no")


def _kit_colors(details: PlayerDetails) -> tuple[str | None, str | None]:
    if jersey_colors_configured(details):
        return details.primaryJerseyColor, details.secondaryJerseyColor
    return None, None


def _rows_without_masks(rows: list[Row]) -> list[Row]:
    return [r.model_copy(update={"mask_rle": None}) for r in rows]


def _rows_for_heatmap(
    rows: list[Row],
    target_track_id: int | None,
    target_jersey: int | None = None,
) -> tuple[list[Row], str | None]:
    """Target-attributed rows (player jersey), primary track, or fallback demo track."""
    if target_jersey is not None and target_jersey > 0:
        attributed = [r for r in rows if r.player == target_jersey]
        if attributed:
            return attributed, "locked_target"
    if target_track_id is not None:
        return [r for r in rows if r.track == target_track_id], "locked_target"
    if not rows:
        return [], None
    best_track = Counter(r.track for r in rows).most_common(1)[0][0]
    return [r for r in rows if r.track == best_track], "fallback_track"


def _pitch_calibration_name() -> str:
    from backend.app.pitch_api import default_pitch_calibration_name

    return default_pitch_calibration_name()


def _calibration_matches_video(
    calibration: PitchCalibration, width: int, height: int
) -> bool:
    cal_w, cal_h = calibration.image_size
    return cal_w == width and cal_h == height


def _resolve_pitch_calibration(
    *,
    calibration_key: str | None,
) -> tuple[PitchCalibration | None, str | None]:
    from backend.app.pitch_api import calibration_keys_for_run

    return resolve_calibration(*calibration_keys_for_run(explicit_key=calibration_key))


def _ball_events_enabled() -> bool:
    return os.environ.get("ENABLE_BALL_EVENTS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _ball_debug_dir() -> str | None:
    """Opt-in directory for annotated target-vs-ball debug frames (BALL_DEBUG_DIR)."""
    raw = os.environ.get("BALL_DEBUG_DIR", "").strip()
    if not raw:
        return None
    os.makedirs(raw, exist_ok=True)
    return raw


def _save_ball_debug(
    *,
    out_dir: str,
    frame,
    frame_idx: int,
    target_foot_px: tuple[float, float] | None,
    target_bbox: list[float] | None,
    ball_det,
    target_m: tuple[float, float] | None,
    ball_m: tuple[float, float] | None,
) -> None:
    """Draw target box/foot + detected ball and append a JSONL row. Diagnostic only."""
    import json

    img = frame.copy()
    if target_bbox is not None:
        x1, y1, x2, y2 = (int(v) for v in target_bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if target_foot_px is not None:
        fx, fy = int(target_foot_px[0]), int(target_foot_px[1])
        cv2.circle(img, (fx, fy), 8, (0, 255, 0), -1)
        cv2.putText(
            img, "TARGET", (fx + 10, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

    dist_px: float | None = None
    if ball_det is not None:
        bx1, by1 = int(ball_det.x1), int(ball_det.y1)
        bx2, by2 = int(ball_det.x2), int(ball_det.y2)
        bcx, bcy = int(ball_det.cx_px), int(ball_det.cy_px)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
        cv2.circle(img, (bcx, bcy), 5, (0, 0, 255), -1)
        cv2.putText(
            img,
            f"BALL {ball_det.conf:.2f}",
            (bx1, max(0, by1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        if target_foot_px is not None:
            dist_px = (
                (target_foot_px[0] - ball_det.cx_px) ** 2
                + (target_foot_px[1] - ball_det.cy_px) ** 2
            ) ** 0.5
            cv2.line(
                img,
                (int(target_foot_px[0]), int(target_foot_px[1])),
                (bcx, bcy),
                (0, 255, 255),
                2,
            )

    dist_m: float | None = None
    if target_m is not None and ball_m is not None:
        dist_m = (
            (target_m[0] - ball_m[0]) ** 2 + (target_m[1] - ball_m[1]) ** 2
        ) ** 0.5

    header = f"f{frame_idx}"
    if dist_px is not None:
        header += f" dpx={dist_px:.0f}"
    if dist_m is not None:
        header += f" dm={dist_m:.1f}"
    cv2.putText(
        img, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
    )
    cv2.imwrite(os.path.join(out_dir, f"f{frame_idx:05d}.jpg"), img)

    rec = {
        "frame": frame_idx,
        "target_px": list(target_foot_px) if target_foot_px is not None else None,
        "ball_px": [ball_det.cx_px, ball_det.cy_px] if ball_det is not None else None,
        "ball_conf": ball_det.conf if ball_det is not None else None,
        "dist_px": dist_px,
        "target_m": list(target_m) if target_m is not None else None,
        "ball_m": list(ball_m) if ball_m is not None else None,
        "dist_m": dist_m,
    }
    with open(os.path.join(out_dir, "frames.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _foot_m_from_norm(
    calibration: PitchCalibration,
    foot_nx: float,
    foot_ny: float,
    width: int,
    height: int,
) -> tuple[float, float] | None:
    try:
        return calibration.pixel_to_meters(foot_nx * width, foot_ny * height)
    except ValueError:
        return None


def _foot_m_from_bbox(
    calibration: PitchCalibration,
    bbox: list[float],
) -> tuple[float, float] | None:
    fx, fy = foot_position_pixels(bbox)
    try:
        return calibration.pixel_to_meters(fx, fy)
    except ValueError:
        return None


def _foot_from_locked_bbox(
    *,
    lock,
    tracks: list,
    frame,
    calibration: PitchCalibration | None,
    last_seg_bbox: list[float] | None,
    reid_prototype,
    reid_extractor,
    width: int,
    height: int,
    target_jersey: int | None = None,
    ocr: object | None = None,
    kit_primary: str | None = None,
    kit_secondary: str | None = None,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Foot pixel/metre from locked track bbox when SAM foot is unavailable."""
    bbox: list[float] | None = list(last_seg_bbox) if last_seg_bbox is not None else None
    if bbox is None:
        resolved = resolve_target_bbox(
            tracks,
            lock.track_id,
            reid_prototype=reid_prototype,
            reid_extractor=reid_extractor,
            frame=frame,
            last_bbox=last_seg_bbox,
            frame_width=width,
            frame_height=height,
            calibration=calibration,
            target_jersey=target_jersey,
            ocr=ocr,
            kit_primary=kit_primary,
            kit_secondary=kit_secondary,
        )
        if resolved is not None:
            _, bbox, _ = resolved
    if bbox is None:
        return None, None
    foot_px = foot_position_pixels(bbox)
    foot_m = _foot_m_from_bbox(calibration, bbox) if calibration is not None else None
    return foot_px, foot_m


def _append_gap_row(
    rows: list[Row],
    *,
    ts: float,
    frame_idx: int,
    track_id: int,
    player: int | None,
    bbox: list[float],
    width: int,
    height: int,
    ocr_conf: float | None,
) -> None:
    """Trajectory sample when SAM cannot run (no visible target bbox)."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    fx, fy = foot_position_pixels(bbox)
    rows.append(
        Row(
            t=round(ts, 3),
            frame=frame_idx,
            track=track_id,
            player=player,
            x=round(cx / width if width > 0 else 0.0, 4),
            y=round(cy / height if height > 0 else 0.0, 4),
            bbox=[round(v, 2) for v in bbox],
            ocr_conf=ocr_conf,
            foot_x=round(fx / width if width > 0 else 0.0, 4),
            foot_y=round(fy / height if height > 0 else 0.0, 4),
            mask_rle=None,
            mask_fallback=True,
            segment_source="gap_fill",
            segment_track_id=track_id,
        )
    )


def _rows_for_analytics(
    rows: list[Row],
    *,
    locked_at_frame: int | None,
) -> list[Row]:
    """Trajectory rows for distance + heatmap (lock → end, no occlusion gap-fill)."""
    out = list(rows)
    if locked_at_frame is not None:
        out = [r for r in out if r.frame >= locked_at_frame]
    return [r for r in out if r.segment_source != "gap_fill"]


def _append_target_row(
    rows: list[Row],
    *,
    ts: float,
    frame_idx: int,
    track_id: int,
    player: int | None,
    bbox: list[float],
    width: int,
    height: int,
    seg_result,
    ocr_conf: float | None,
) -> None:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    nx = cx / width if width > 0 else 0.0
    ny = cy / height if height > 0 else 0.0
    rows.append(
        Row(
            t=round(ts, 3),
            frame=frame_idx,
            track=track_id,
            player=player,
            x=round(nx, 4),
            y=round(ny, 4),
            bbox=[round(v, 2) for v in bbox],
            ocr_conf=ocr_conf,
            foot_x=seg_result.foot_nx,
            foot_y=seg_result.foot_ny,
            mask_rle=seg_result.mask_rle,
            mask_fallback=seg_result.mask_fallback,
            segment_source=seg_result.segment_source,
            segment_track_id=seg_result.segment_track_id,
        )
    )


def _read_and_observe_jersey_number(
    matcher: TrackIdentityMatcher,
    ocr: JerseyOCR,
    *,
    frame,
    bbox: list[float],
    track_id: int,
    target_jersey: int,
    frame_idx: int,
    embedding,
    ocr_every: int = 1,
) -> tuple[int | None, float, list[tuple[str, float]]]:
    """One jersey read per track per frame; weighted votes when classifier is loaded."""
    reader = ocr._number_reader
    if reader.available:
        num, conf, clf_num, clf_conf, ocr_num, ocr_conf, raw = (
            ocr.read_number_sources_detailed(
                frame, bbox, target_number=target_jersey
            )
        )
        matcher.observe_number(
            track_id,
            num,
            conf,
            frame_idx=frame_idx,
            ocr_every=ocr_every,
            classifier_number=clf_num,
            classifier_conf=clf_conf,
            ocr_number=ocr_num,
            ocr_conf=ocr_conf,
            use_classifier_weighted_vote=True,
            embedding=embedding,
        )
        return num, conf, raw
    num, conf, raw = ocr.read_number_detailed(
        frame, bbox, target_number=target_jersey
    )
    matcher.observe_number(
        track_id,
        num,
        conf,
        frame_idx=frame_idx,
        ocr_every=ocr_every,
        embedding=embedding,
    )
    return num, conf, raw


def _scan_jersey_for_ball_events(
    *,
    frame,
    frame_idx: int,
    tracks: list,
    track_frame_count: dict[int, int],
    track_min_age: int,
    ocr: JerseyOCR,
    target_jersey: int,
    ocr_every: int,
    last_ocr_frame: dict[int, int],
    frame_ocr_match: tuple[int, float] | None,
) -> tuple[int, float] | None:
    """Extra jersey OCR pass for ball-event soft ID (pre-lock windows).

    The identity ``analyze`` path can miss frames between OCR intervals; this
    scans mature tracks when we still need a ball-foot anchor before hard lock.
    """
    if frame_ocr_match is not None or target_jersey <= 0:
        return frame_ocr_match
    if frame_idx % ocr_every != 0:
        return frame_ocr_match
    from backend.app.pipeline.segmentation import seg_jersey_ocr_min_conf

    min_conf = seg_jersey_ocr_min_conf()
    best: tuple[int, float] | None = None
    for tr in tracks:
        track_id = int(tr["track_id"])
        if track_frame_count.get(track_id, 0) < track_min_age:
            continue
        if frame_idx - last_ocr_frame.get(track_id, -ocr_every) < ocr_every:
            continue
        bbox = tr["bbox"]
        num, num_conf, _ = ocr.read_number_detailed(
            frame,
            bbox,
            target_number=target_jersey,
        )
        last_ocr_frame[track_id] = frame_idx
        if num == target_jersey and num_conf >= min_conf:
            if best is None or num_conf > best[1]:
                best = (track_id, num_conf)
    return best


def run_pipeline(
    video_path: str,
    details: PlayerDetails,
    *,
    calibration_key: str | None = None,
    include_masks_in_response: bool = False,
    output_video_path: str | None = None,
) -> AnalyzeResponse:
    """Run legacy analyze. Set include_masks_in_response=True for CLI mask export (keeps mask_rle when API would strip).

    When output_video_path is set, writes an annotated MP4 (track ellipses, lock highlight, ball marker).
    """
    detector = get_person_detector()
    tracker = PlayerTracker()
    ocr = JerseyOCR()
    matcher = TrackIdentityMatcher(
        target_jersey=details.jerseyNumber,
        target_name=details.name,
    )
    reid_extractor = get_reid_extractor()
    prepare_mobile_sam_for_run()
    segmenter = get_mobile_sam_segmenter()
    seg_state = SegmentationState()
    ocr_debug = OcrDebug()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_writer: cv2.VideoWriter | None = None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        rows: list[Row] = []
        frame_idx = 0
        ocr_every = _ocr_every()
        track_min_age = _track_min_age()
        last_ocr_frame: dict[int, int] = defaultdict(lambda: -ocr_every)
        last_ocr_conf: dict[int, float] = {}
        track_frame_count: dict[int, int] = defaultdict(int)
        locked_at_frame: int | None = None
        segmentation_started_at_frame: int | None = None
        last_sam_frame: int | None = None
        last_seg_bbox: list[float] | None = None
        frame_cap = _max_frames()
        frame_start = _frame_start()
        kit_primary, kit_secondary = _kit_colors(details)

        logger.info(
            "Pipeline processing: video=%s frames_cap=%s jersey=%s calibration=%s",
            video_path,
            frame_cap,
            details.jerseyNumber,
            calibration_key or "(default)",
        )

        calibration_skipped_reason: str | None = None
        calibration, matched_cal_name = _resolve_pitch_calibration(
            calibration_key=calibration_key,
        )
        if calibration is not None and not _calibration_matches_video(
            calibration, width, height
        ):
            calibration_skipped_reason = "size_mismatch"
            calibration = None
            matched_cal_name = None

        ball_detector = None
        ball_states: list = []
        target_tracker = None
        target_samples: list = []
        last_anchor_foot_px: tuple[float, float] | None = None
        this_frame_foot_m: tuple[float, float] | None = None
        this_frame_foot_px: tuple[float, float] | None = None
        this_frame_target_visible = False
        this_frame_seg_source: str | None = None
        enable_ball = _ball_events_enabled()
        ball_debug_dir = _ball_debug_dir()
        scene_detector = SceneCutDetector()
        scene_cut_count = 0
        suppressed_frame_count = 0
        # Lock windows bounded by scene cuts; archived (not discarded) on each cut so
        # the best/longest epoch is analysed instead of whatever happens to be last.
        epochs: list = []
        epoch_start_frame = frame_start
        ball_carry_track_id: int | None = None

        if output_video_path:
            video_writer = open_browser_video_writer(
                output_video_path, fps, width, height
            )

        while frame_idx < frame_start:
            ok, _ = cap.read()
            if not ok:
                break
            frame_idx += 1

        while True:
            if frame_idx >= frame_start + frame_cap:
                break
            ok, frame = cap.read()
            if not ok:
                break

            # Broadcast cut: invalidate cross-shot tracker/lock state and drop the
            # timeline collected for the prior shot before processing this frame.
            if scene_detector.update(frame):
                scene_cut_count += 1
                from backend.app.pipeline.ball_events.run_events import LockEpoch

                _prev_lock = matcher.lock
                epochs.append(
                    LockEpoch(
                        start_frame=epoch_start_frame,
                        end_frame=frame_idx,
                        locked_at_frame=locked_at_frame,
                        track_id=_prev_lock.track_id if _prev_lock else None,
                        ball_states=ball_states,
                        target_samples=target_samples,
                    )
                )
                epoch_start_frame = frame_idx
                tracker.reset()
                matcher.release_lock()
                seg_state = SegmentationState()
                locked_at_frame = None
                segmentation_started_at_frame = None
                last_sam_frame = None
                last_seg_bbox = None
                last_anchor_foot_px = None
                if ball_detector is not None:
                    ball_detector.reset()
                if target_tracker is not None:
                    target_tracker.reset()
                ball_states = []
                target_samples = []
                ball_carry_track_id = None
                logger.info("Scene cut at frame %s — tracker/lock reset", frame_idx)

            skip_ball = scene_detector.is_suppressed
            if skip_ball:
                suppressed_frame_count += 1

            matcher.begin_frame(frame_idx, ocr_every=ocr_every)

            ts = frame_idx / fps if fps > 0 else 0.0
            this_frame_foot_m = None
            this_frame_foot_px = None
            this_frame_target_visible = False
            this_frame_seg_source = None
            frame_ocr_match: tuple[int, float] | None = None
            ball_bbox: list[float] | None = None

            dets = detector(frame)
            tracks = tracker.update(dets, frame)
            for tr in tracks:
                track_id = tr["track_id"]
                track_frame_count[track_id] += 1
                bbox = tr["bbox"]
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                nx = cx / width if width > 0 else 0.0
                ny = cy / height if height > 0 else 0.0

                frame_lock = matcher.lock
                is_locked_track = (
                    frame_lock is not None and frame_lock.track_id == track_id
                )
                mature = track_frame_count[track_id] >= track_min_age
                analyze = (
                    mature
                    and not matcher.is_locked
                    and is_valid_bbox(bbox, frame.shape, min_size=16)
                )

                embedding = None
                if reid_extractor is not None and (
                    analyze or (matcher.lock and matcher.lock.track_id == track_id)
                ):
                    kit_crop = JerseyOCR.full_kit_crop(frame, bbox)
                    embedding = reid_extractor.embed(kit_crop)
                    matcher.observe_reid(track_id, embedding)

                if analyze:
                    color_score = 0.0
                    if jersey_colors_configured(details):
                        color_score = jersey_color_score(
                            frame,
                            bbox,
                            details.primaryJerseyColor,
                            details.secondaryJerseyColor,
                        )
                        matcher.observe_color(track_id, color_score)

                    should_ocr = frame_idx - last_ocr_frame[track_id] >= ocr_every
                    if should_ocr:
                        last_ocr_frame[track_id] = frame_idx
                        num, num_conf, num_raw = _read_and_observe_jersey_number(
                            matcher,
                            ocr,
                            frame=frame,
                            bbox=bbox,
                            track_id=track_id,
                            target_jersey=details.jerseyNumber,
                            frame_idx=frame_idx,
                            embedding=embedding,
                            ocr_every=ocr_every,
                        )
                        name, name_conf, name_raw = ocr.read_name_detailed(frame, bbox)
                        matcher.observe_name(track_id, name, name_conf)
                        if num_conf > 0:
                            last_ocr_conf[track_id] = num_conf
                        from backend.app.pipeline.segmentation import (
                            seg_jersey_ocr_min_conf,
                        )

                        if (
                            num == details.jerseyNumber
                            and num_conf >= seg_jersey_ocr_min_conf()
                            and details.jerseyNumber > 0
                        ):
                            if (
                                frame_ocr_match is None
                                or num_conf > frame_ocr_match[1]
                            ):
                                frame_ocr_match = (track_id, num_conf)

                        crop_path = ocr_debug.save_crop(
                            frame_idx,
                            track_id,
                            JerseyOCR.full_kit_crop(frame, bbox),
                        )
                        ocr_debug.log(
                            frame_idx=frame_idx,
                            track_id=track_id,
                            number_raw=num_raw,
                            parsed_number=num,
                            number_conf=num_conf,
                            name_raw=name_raw,
                            parsed_name=name,
                            name_conf=name_conf,
                            color_score=color_score,
                            crop_path=crop_path,
                        )

                        new_lock = matcher.try_acquire_lock()
                        if new_lock is not None:
                            if locked_at_frame is None:
                                locked_at_frame = frame_idx
                            seg_state.acquire_lock(new_lock.track_id)
                            if track_id == new_lock.track_id:
                                seg_state.record_visible_frame(frame_idx)
                        elif matcher.lock is not None:
                            matcher.maybe_upgrade_lock(track_id)

                player = (
                    details.jerseyNumber
                    if is_locked_track and details.jerseyNumber > 0
                    else None
                )

                if matcher.is_locked and seg_state.sam_active:
                    continue

                if matcher.is_locked and frame_lock is not None:
                    if not is_locked_track or not mature:
                        continue
                elif not mature and not is_locked_track:
                    continue

                rows.append(
                    Row(
                        t=round(ts, 3),
                        frame=frame_idx,
                        track=track_id,
                        player=player,
                        x=round(nx, 4),
                        y=round(ny, 4),
                        bbox=[round(v, 2) for v in bbox],
                        ocr_conf=last_ocr_conf.get(track_id),
                    )
                )

            # Warmup: target visible if locked track OR ReID fallback finds them.
            lock = matcher.lock
            if enable_ball and not skip_ball and lock is None:
                frame_ocr_match = _scan_jersey_for_ball_events(
                    frame=frame,
                    frame_idx=frame_idx,
                    tracks=tracks,
                    track_frame_count=track_frame_count,
                    track_min_age=track_min_age,
                    ocr=ocr,
                    target_jersey=details.jerseyNumber,
                    ocr_every=ocr_every,
                    last_ocr_frame=last_ocr_frame,
                    frame_ocr_match=frame_ocr_match,
                )
            if lock is not None:
                seg_state.acquire_lock(lock.track_id)
                target_visible = resolve_target_bbox(
                    tracks,
                    lock.track_id,
                    reid_prototype=matcher.reid_prototype,
                    reid_extractor=reid_extractor,
                    frame=frame,
                    last_bbox=last_seg_bbox,
                    frame_width=width,
                    frame_height=height,
                    calibration=calibration,
                    target_jersey=details.jerseyNumber,
                    ocr=ocr,
                    kit_primary=kit_primary,
                    kit_secondary=kit_secondary,
                )
                if target_visible is not None:
                    seg_state.record_visible_frame(frame_idx)
                elif locked_at_frame == frame_idx:
                    # OCR/ReID lock can fire while the locked track_id is not in
                    # this frame's detections; still count the lock frame for warmup.
                    seg_state.record_visible_frame(frame_idx)
                elif not seg_state.sam_active:
                    seg_state.record_miss_frame()

            if lock is not None and seg_state.sam_active:
                resolved = resolve_target_bbox(
                    tracks,
                    lock.track_id,
                    reid_prototype=matcher.reid_prototype,
                    reid_extractor=reid_extractor,
                    frame=frame,
                    last_bbox=last_seg_bbox,
                    frame_width=width,
                    frame_height=height,
                    calibration=calibration,
                    target_jersey=details.jerseyNumber,
                    ocr=ocr,
                    kit_primary=kit_primary,
                    kit_secondary=kit_secondary,
                )
                if resolved is not None:
                    seg_tid, seg_bbox, is_reid_fb = resolved
                    if should_run_sam_this_frame(frame_idx, last_sam_frame):
                        seg_result = run_segmentation_for_target(
                            frame,
                            seg_bbox,
                            seg_tid,
                            is_reid_fallback=is_reid_fb,
                            width=width,
                            height=height,
                            segmenter=segmenter,
                        )
                        last_sam_frame = frame_idx
                    else:
                        seg_result = segment_result_from_bbox(
                            seg_bbox, seg_tid, width, height
                        )
                    if segmentation_started_at_frame is None:
                        segmentation_started_at_frame = (
                            seg_state.segmentation_started_at_frame
                        )

                    ocr_bbox = (
                        mask_bbox(seg_result.mask)
                        if seg_result.mask is not None
                        else seg_bbox
                    )
                    if (
                        is_valid_bbox(ocr_bbox, frame.shape, min_size=16)
                        and frame_idx - last_ocr_frame[seg_tid] >= ocr_every
                    ):
                        last_ocr_frame[seg_tid] = frame_idx
                        num, num_conf, num_raw = _read_and_observe_jersey_number(
                            matcher,
                            ocr,
                            frame=frame,
                            bbox=ocr_bbox,
                            track_id=seg_tid,
                            target_jersey=details.jerseyNumber,
                            frame_idx=frame_idx,
                            embedding=(
                                reid_extractor.embed(
                                    JerseyOCR.crop_for_ocr(frame, seg_bbox, ocr_bbox)
                                )
                                if reid_extractor is not None
                                else None
                            ),
                            ocr_every=ocr_every,
                        )
                        name, name_conf, name_raw = ocr.read_name_detailed(
                            frame, ocr_bbox
                        )
                        matcher.observe_name(seg_tid, name, name_conf)
                        if num_conf > 0:
                            last_ocr_conf[seg_tid] = num_conf
                        matcher.maybe_upgrade_lock(seg_tid)
                        from backend.app.pipeline.segmentation import (
                            seg_jersey_ocr_min_conf,
                        )

                        if (
                            num == details.jerseyNumber
                            and num_conf >= seg_jersey_ocr_min_conf()
                            and details.jerseyNumber > 0
                        ):
                            if (
                                frame_ocr_match is None
                                or num_conf > frame_ocr_match[1]
                            ):
                                frame_ocr_match = (seg_tid, num_conf)

                    player_num = (
                        details.jerseyNumber if details.jerseyNumber > 0 else None
                    )
                    _append_target_row(
                        rows,
                        ts=ts,
                        frame_idx=frame_idx,
                        track_id=seg_tid,
                        player=player_num,
                        bbox=seg_bbox,
                        width=width,
                        height=height,
                        seg_result=seg_result,
                        ocr_conf=last_ocr_conf.get(seg_tid),
                    )
                    last_seg_bbox = list(seg_bbox)
                    if calibration is not None:
                        this_frame_foot_px = (
                            seg_result.foot_nx * width,
                            seg_result.foot_ny * height,
                        )
                        this_frame_foot_m = _foot_m_from_norm(
                            calibration,
                            seg_result.foot_nx,
                            seg_result.foot_ny,
                            width,
                            height,
                        )
                        this_frame_target_visible = True
                        this_frame_seg_source = seg_result.segment_source
                elif last_seg_bbox is not None and details.jerseyNumber > 0:
                    # Target was not positively located this frame (no jersey/kit/ReID
                    # match). Continue the trajectory from the last good bbox, but DO NOT
                    # segment: prompting MobileSAM with a stale box segments whatever now
                    # occupies that spot (a shadow, grass, another person) and paints a
                    # ghost overlay even though the player has left the frame.
                    _append_gap_row(
                        rows,
                        ts=ts,
                        frame_idx=frame_idx,
                        track_id=lock.track_id,
                        player=details.jerseyNumber,
                        bbox=last_seg_bbox,
                        width=width,
                        height=height,
                        ocr_conf=last_ocr_conf.get(lock.track_id),
                    )
                    this_frame_seg_source = "gap_fill"
                    if this_frame_foot_px is None:
                        this_frame_foot_px = foot_position_pixels(last_seg_bbox)
                elif (
                    calibration is not None
                    and lock is not None
                    and this_frame_foot_m is None
                ):
                    resolved = resolve_target_bbox(
                        tracks,
                        lock.track_id,
                        reid_prototype=matcher.reid_prototype,
                        reid_extractor=reid_extractor,
                        frame=frame,
                        last_bbox=last_seg_bbox,
                        frame_width=width,
                        frame_height=height,
                        calibration=calibration,
                        target_jersey=details.jerseyNumber,
                        ocr=ocr,
                        kit_primary=kit_primary,
                        kit_secondary=kit_secondary,
                    )
                    if resolved is not None:
                        _, bbox, _ = resolved
                        fx, fy = foot_position_pixels(bbox)
                        this_frame_foot_px = (fx, fy)
                        this_frame_foot_m = _foot_m_from_bbox(calibration, bbox)
                        this_frame_target_visible = True

            if enable_ball and not skip_ball:
                if ball_detector is None:
                    from backend.app.pipeline.ball_events.ball_detector import (
                        BallDetector,
                    )
                    from backend.app.pipeline.ball_events.timeline import (
                        TargetFootTracker,
                    )

                    ball_detector = BallDetector(fps=fps)
                    target_tracker = TargetFootTracker()
                ball_id_type: str | None = "locked" if lock is not None else None
                if lock is not None and this_frame_foot_px is None:
                    foot_px, foot_m = _foot_from_locked_bbox(
                        lock=lock,
                        tracks=tracks,
                        frame=frame,
                        calibration=calibration,
                        last_seg_bbox=last_seg_bbox,
                        reid_prototype=matcher.reid_prototype,
                        reid_extractor=reid_extractor,
                        width=width,
                        height=height,
                        target_jersey=details.jerseyNumber,
                        ocr=ocr,
                        kit_primary=kit_primary,
                        kit_secondary=kit_secondary,
                    )
                    if foot_px is not None:
                        this_frame_foot_px = foot_px
                        if this_frame_foot_m is None:
                            this_frame_foot_m = foot_m
                        if this_frame_seg_source is None:
                            this_frame_seg_source = "bbox_track"
                            this_frame_target_visible = True
                elif lock is None and calibration is not None:
                    from backend.app.pipeline.ball_events.target_resolve import (
                        resolve_ball_event_target,
                    )

                    ball_obs, ball_carry_track_id = resolve_ball_event_target(
                        tracks=tracks,
                        matcher=matcher,
                        lock=None,
                        calibration=calibration,
                        frame_shape=frame.shape,
                        frame_ocr_match=frame_ocr_match,
                        carry_track_id=ball_carry_track_id,
                    )
                    if ball_obs is not None:
                        this_frame_foot_px = ball_obs.foot_px
                        this_frame_foot_m = ball_obs.foot_m
                        this_frame_target_visible = ball_obs.target_visible
                        this_frame_seg_source = ball_obs.segment_source
                        ball_id_type = ball_obs.identification_type
                if ball_detector.available:
                    anchor_px = this_frame_foot_px or last_anchor_foot_px
                    raw = ball_detector.detect(frame, near_px=anchor_px)
                    if raw is not None:
                        ball_bbox = [raw.x1, raw.y1, raw.x2, raw.y2]
                    if this_frame_foot_px is not None:
                        last_anchor_foot_px = this_frame_foot_px
                    if calibration is not None:
                        ball_states.append(
                            ball_detector.update_pitch(frame_idx, raw, calibration)
                        )
                    else:
                        ball_states.append(
                            ball_detector.update_pixels(frame_idx, raw)
                        )
                    if (
                        ball_debug_dir is not None
                        and this_frame_foot_px is not None
                        and raw is not None
                    ):
                        _ball_state = ball_states[-1]
                        _save_ball_debug(
                            out_dir=ball_debug_dir,
                            frame=frame,
                            frame_idx=frame_idx,
                            target_foot_px=this_frame_foot_px,
                            target_bbox=last_seg_bbox,
                            ball_det=raw,
                            target_m=this_frame_foot_m,
                            ball_m=_ball_state.pitch_m or _ball_state.smooth_m,
                        )
                if target_tracker is not None and calibration is not None:
                    if lock is not None or ball_id_type is not None:
                        target_samples.append(
                            target_tracker.observe(
                                frame_idx,
                                this_frame_foot_m,
                                visible=this_frame_target_visible,
                                segment_source=this_frame_seg_source,
                                foot_px=this_frame_foot_px,
                                identification_type=ball_id_type,
                            )
                        )

            if video_writer is not None:
                frame_lock = matcher.lock
                locked_id = frame_lock.track_id if frame_lock is not None else None
                track_draws = [
                    TrackDraw(
                        track_id=tr["track_id"],
                        bbox=tr["bbox"],
                        confidence=tr.get("score", 1.0),
                        label=(
                            str(details.jerseyNumber)
                            if locked_id == tr["track_id"] and details.jerseyNumber > 0
                            else None
                        ),
                    )
                    for tr in tracks
                ]
                annotated = draw_tracks(
                    frame,
                    track_draws,
                    locked_track_id=locked_id,
                    ball_bbox=ball_bbox,
                )
                video_writer.write(annotated)

            frame_idx += 1
            if frame_idx % 250 == 0:
                logger.info(
                    "Pipeline progress: frame %s/%s lock=%s method=%s",
                    frame_idx,
                    min(frame_start + frame_cap, total_frames or frame_start + frame_cap),
                    matcher.lock.track_id if matcher.lock else None,
                    matcher.lock.method if matcher.lock else "none",
                )

        # Archive the final in-progress lock window so its samples are analysed too.
        from backend.app.pipeline.ball_events.run_events import LockEpoch

        _final_lock = matcher.lock
        epochs.append(
            LockEpoch(
                start_frame=epoch_start_frame,
                end_frame=frame_idx,
                locked_at_frame=locked_at_frame,
                track_id=_final_lock.track_id if _final_lock else None,
                ball_states=ball_states,
                target_samples=target_samples,
            )
        )

        if sam_enabled() and not seg_state.sam_active:
            logger.warning(
                "MobileSAM never activated: locked_track_id=%s visible_count=%s "
                "warmup_needed=%s frames_processed=%s segmenter_loaded=%s",
                seg_state.lock_track_id,
                seg_state.visible_count,
                sam_warmup_frames(),
                frame_idx,
                segmenter is not None,
            )

        if frame_idx == 0 and total_frames == 0:
            total_frames = frame_idx

        if scene_cut_count:
            logger.info(
                "Scene cuts: %s detected, %s frames had ball events suppressed",
                scene_cut_count,
                suppressed_frame_count,
            )

        duration_s = frame_idx / fps if fps > 0 else 0.0
        final_lock = matcher.finalize()

        if final_lock is not None and details.jerseyNumber > 0:
            target_id = final_lock.track_id
            rows = [
                row.model_copy(update={"player": details.jerseyNumber})
                if row.player is None
                else row
                for row in rows
            ]
            target = TargetMatch(
                jersey=details.jerseyNumber,
                track_id=target_id,
                confidence=final_lock.confidence,
                method=final_lock.method,
                locked_at_frame=locked_at_frame,
                segmentation_started_at_frame=segmentation_started_at_frame
                or seg_state.segmentation_started_at_frame,
            )
        else:
            target = TargetMatch(
                jersey=details.jerseyNumber,
                track_id=None,
                confidence=0.0,
                method="none",
                locked_at_frame=None,
                segmentation_started_at_frame=None,
            )

        movement: MovementStatsSchema | None = None
        heatmap: HeatmapResult | None = None
        calibration_name: str | None = matched_cal_name if calibration else None

        event_counts: PlayerEventCounts | None = None
        inferred_events: list[InferredBallEvent] | None = None
        ball_samples: int | None = None
        events_unavailable_reason: str | None = None
        drive_contact_m: float | None = None
        provenance: str | None = None
        best_epoch = None

        if enable_ball:
            if ball_detector is None or not ball_detector.available:
                events_unavailable_reason = "no_ball_weights"
            elif calibration is None:
                events_unavailable_reason = "no_calibration"
            else:
                from backend.app.pipeline.ball_events.run_events import (
                    _epoch_analyzable,
                    run_events_multi,
                )

                min_visible = int(
                    os.environ.get("EPOCH_MIN_VISIBLE_FRAMES", "5") or "5"
                )
                if not any(
                    _epoch_analyzable(ep, min_visible=min_visible) for ep in epochs
                ):
                    events_unavailable_reason = "no_lock"
                else:
                    event_counts, ball_samples, best_epoch, detected, drive_contact_m = (
                        run_events_multi(epochs, fps=fps)
                    )
                    inferred_events = [
                        InferredBallEvent(
                            frame=ev.frame,
                            kind=ev.kind,
                            lock_confidence=ev.lock_confidence,  # type: ignore[arg-type]
                            detection_phase=ev.detection_phase,
                            possession_confidence=ev.confidence,
                        )
                        for ev in detected
                    ]
                    provenance = "inferred"
        elif not enable_ball:
            events_unavailable_reason = None

        # Heatmap / movement are single-track artefacts that cannot merge across
        # re-locks, so scope them to the best lock epoch when one was analysed.
        if best_epoch is not None and best_epoch.track_id is not None:
            hm_rows_src = [
                r
                for r in rows
                if best_epoch.start_frame <= r.frame < best_epoch.end_frame
            ]
            hm_track_id = best_epoch.track_id
            hm_locked_at = best_epoch.locked_at_frame
        else:
            hm_rows_src = rows
            hm_track_id = target.track_id
            hm_locked_at = target.locked_at_frame

        heatmap_rows, heatmap_source = _rows_for_heatmap(
            hm_rows_src,
            hm_track_id,
            target_jersey=details.jerseyNumber if hm_track_id else None,
        )

        analytics_rows = _rows_for_analytics(
            heatmap_rows,
            locked_at_frame=hm_locked_at,
        )

        seg_start = target.segmentation_started_at_frame
        if seg_start is not None:
            heatmap_rows = [r for r in heatmap_rows if r.frame >= seg_start]

        keep_masks = include_masks_in_response or _include_masks_in_api_response()
        if include_masks_in_response:
            # Overlay export: keep every SAM row, not only heatmap-scoped track rows.
            api_rows = [r for r in rows if r.mask_rle is not None]
        elif keep_masks:
            api_rows = heatmap_rows
        else:
            api_rows = _rows_without_masks(heatmap_rows)

        mask_row_count = sum(1 for r in heatmap_rows if r.mask_rle is not None)
        masks_available = mask_row_count > 0
        masks_unavailable_reason: str | None = None
        if not masks_available and segmenter is None and sam_enabled():
            masks_unavailable_reason = mobile_sam_unavailable_reason() or (
                "MobileSAM did not load — install timm and mobile_sam in the backend "
                "venv (see backend/weights/README.md), then re-run Analyze."
            )
        elif (
            not masks_available
            and target.segmentation_started_at_frame is not None
            and segmenter is not None
        ):
            masks_unavailable_reason = (
                "MobileSAM ran but produced no masks for this video — check SAM_DEVICE "
                "or backend logs."
            )
        elif not masks_available and target.method == "none":
            masks_unavailable_reason = (
                "No player lock — enter the correct jersey number (and optional name/"
                "jersey colors), then re-run Analyze."
            )
        elif not masks_available and target.method == "color":
            masks_unavailable_reason = (
                "Only jersey-color lock matched (OCR did not confirm the number). "
                "Set the correct jersey number and primary/secondary kit colors, "
                "then re-run Analyze."
            )
        elif not masks_available and target.locked_at_frame is not None:
            warmup = sam_warmup_frames()
            masks_unavailable_reason = (
                f"Player locked at frame {target.locked_at_frame} but MobileSAM "
                f"warmup ({warmup} consecutive visible frames) did not finish — "
                "keep the target in view after lock or lower SAM_WARMUP_FRAMES."
            )
        elif not masks_available and target.segmentation_started_at_frame is None:
            masks_unavailable_reason = (
                "Player lock or SAM warmup did not complete — verify jersey number/"
                "OCR lock, then re-run Analyze."
            )

        if len(analytics_rows) >= 2 and calibration is not None:
            stats = compute_movement_stats_from_rows(
                analytics_rows, fps=fps, calibration=calibration
            )
            if stats.units == "meters":
                movement = MovementStatsSchema(**stats.to_dict())
        if analytics_rows and calibration is not None:
            hm = build_heatmap_from_rows(analytics_rows, calibration, fps=fps)
            heatmap = HeatmapResult(**hm.to_dict())
            if hm.sample_count == 0 and calibration_skipped_reason is None:
                calibration_skipped_reason = "positions_out_of_bounds"

        return AnalyzeResponse(
            video=VideoMeta(
                fps=round(fps, 3),
                width=width,
                height=height,
                frames=frame_idx,
                duration_s=round(duration_s, 3),
                frame_cap=frame_cap if frame_cap < UNLIMITED_FRAME_CAP else None,
                frame_start=frame_start,
            ),
            target=target,
            rows=api_rows,
            movement=movement,
            heatmap=heatmap,
            calibration_name=calibration_name,
            heatmap_source=heatmap_source if heatmap is not None else None,
            calibration_skipped_reason=calibration_skipped_reason,
            masks_available=masks_available,
            masks_unavailable_reason=masks_unavailable_reason,
            mask_row_count=mask_row_count,
            event_counts=event_counts,
            inferred_events=inferred_events,
            provenance=provenance,  # type: ignore[arg-type]
            ball_samples=ball_samples,
            events_unavailable_reason=events_unavailable_reason,
            drive_contact_m=drive_contact_m,
        )
    finally:
        if video_writer is not None:
            video_writer.release()
            if output_video_path:
                out = Path(output_video_path)
                if not ensure_browser_playable_mp4(out):
                    logger.warning(
                        "Annotated video is not browser-playable (install ffmpeg): %s",
                        out,
                    )
        cap.release()
        _release_inference_memory()
