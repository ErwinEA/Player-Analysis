#!/usr/bin/env python3
"""Render segmentation overlay for a named player (jersey / kit color / ReID).

Unlike the full analyze pipeline, this does not wait for matcher lock or SAM
warmup — every frame it picks the best target bbox (jersey OCR, then kit color,
then ReID) and runs MobileSAM when a player is found.

Example (repo root, venv active):

  python backend/scripts/render_player_seg_overlay.py \\
    backend/data/pitch_calibration/testmatch2.mp4 \\
    backend/data/lozano-details.json \\
    -o backend/data/outputs/seg_overlay_lozano.mp4 \\
    --max-frames 1500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from backend.app.pipeline.detector import PersonDetector
from backend.app.pipeline.ocr import JerseyOCR
from backend.app.pipeline.reid import get_reid_extractor
from backend.app.pipeline.run import _kit_colors
from backend.app.pipeline.segmentation import (
    get_mobile_sam_segmenter,
    prepare_mobile_sam_for_run,
    resolve_target_bbox,
    run_segmentation_for_target,
)
from backend.app.pipeline.tracker import PlayerTracker
from backend.app.schemas import PlayerDetails, Row
from backend.tools.render_mask_overlay_video import _composite_mask, render


def _tracks_from_tracker(tracker_out: list) -> list[dict]:
    return [{"track_id": int(t["track_id"]), "bbox": list(t["bbox"])} for t in tracker_out]


def main() -> int:
    parser = argparse.ArgumentParser(description="Player-targeted SAM overlay video")
    parser.add_argument("video", type=Path)
    parser.add_argument("details", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=1500)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument(
        "--ocr-every",
        type=int,
        default=8,
        help="Run jersey OCR on candidates every N frames (color+ReID other frames)",
    )
    args = parser.parse_args()

    if not args.video.is_file() or not args.details.is_file():
        print("Video or details JSON not found", file=sys.stderr)
        return 1

    details = PlayerDetails.model_validate(
        json.loads(args.details.read_text(encoding="utf-8"))
    )
    kit_primary, kit_secondary = _kit_colors(details)

    prepare_mobile_sam_for_run()
    segmenter = get_mobile_sam_segmenter()
    if segmenter is None:
        print("MobileSAM failed to load", file=sys.stderr)
        return 1

    detector = PersonDetector()
    tracker = PlayerTracker()
    ocr = JerseyOCR()
    reid = get_reid_extractor()
    prototype: np.ndarray | None = None

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Could not open {args.video}", file=sys.stderr)
        return 1

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    start = max(0, args.frame_start)
    end = start + max(1, args.max_frames)

    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    rows: list[Row] = []
    last_lock_tid = -1
    frame_idx = start

    while frame_idx < end:
        ok, frame = cap.read()
        if not ok:
            break

        dets = detector(frame)
        tracks = _tracks_from_tracker(tracker.update(dets, frame))

        lock_tid = last_lock_tid
        if tracks:
            if lock_tid not in {t["track_id"] for t in tracks}:
                lock_tid = tracks[0]["track_id"]

        resolved = None
        if tracks and lock_tid >= 0:
            ocr_this_frame = ocr if args.ocr_every <= 1 or frame_idx % args.ocr_every == 0 else None
            resolved = resolve_target_bbox(
                tracks,
                lock_tid,
                reid_prototype=prototype,
                reid_extractor=reid,
                frame=frame,
                last_bbox=None,
                frame_width=width,
                frame_height=height,
                calibration=None,
                target_jersey=details.jerseyNumber,
                ocr=ocr_this_frame,
                kit_primary=kit_primary,
                kit_secondary=kit_secondary,
            )

        if resolved is not None:
            seg_tid, seg_bbox, _is_fb = resolved
            last_lock_tid = seg_tid
            if prototype is None and reid is not None:
                num, conf, _ = ocr.read_number_detailed(
                    frame, seg_bbox, target_number=details.jerseyNumber
                )
                if num == details.jerseyNumber and conf >= 0.35:
                    emb = reid.embed(JerseyOCR.full_kit_crop(frame, seg_bbox))
                    vec = np.asarray(emb, dtype=np.float32).reshape(-1)
                    n = np.linalg.norm(vec)
                    if n > 0:
                        prototype = vec / n
            seg = run_segmentation_for_target(
                frame,
                seg_bbox,
                seg_tid,
                is_reid_fallback=False,
                width=width,
                height=height,
                segmenter=segmenter,
            )
            if seg.mask_rle is not None:
                rows.append(
                    Row(
                        t=round(frame_idx / fps, 3),
                        frame=frame_idx,
                        track=seg_tid,
                        player=details.jerseyNumber,
                        x=seg.foot_nx,
                        y=seg.foot_ny,
                        bbox=[round(v, 2) for v in seg_bbox],
                        foot_x=seg.foot_nx,
                        foot_y=seg.foot_ny,
                        mask_rle=seg.mask_rle,
                        mask_fallback=seg.mask_fallback,
                        segment_source=seg.segment_source,
                        segment_track_id=seg_tid,
                    )
                )

        frame_idx += 1

    cap.release()
    print(f"Collected {len(rows)} frames with SAM masks (frames {start}-{frame_idx - 1})")
    if not rows:
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    return render(
        args.video,
        rows,
        fps,
        args.output,
        frame_start=start,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    raise SystemExit(main())
