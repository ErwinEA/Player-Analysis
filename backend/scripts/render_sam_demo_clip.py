#!/usr/bin/env python3
"""Render a short SAM overlay clip without the full analyze lock path.

Useful to verify MobileSAM visually when pipeline lock is late or missing.
Picks the largest person detection on the anchor frame, then segments that bbox
each frame for the clip length.

Example (repo root, venv active):

  python backend/scripts/render_sam_demo_clip.py \\
    backend/data/pitch_calibration/testmatch2.mp4 \\
    -o backend/data/outputs/sam_demo_clip.mp4 \\
    --anchor-frame 400 --num-frames 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from backend.app.pipeline.detector import PersonDetector
from backend.app.pipeline.segmentation import (
    get_mobile_sam_segmenter,
    prepare_mobile_sam_for_run,
    run_segmentation_for_target,
)
from backend.tools.render_mask_overlay_video import _composite_mask


def _largest_bbox(detections: list) -> list[float] | None:
    best: list[float] | None = None
    best_area = 0.0
    for bbox, _score in detections:
        x1, y1, x2, y2 = bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > best_area:
            best_area = area
            best = list(bbox)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM demo overlay (no player lock).")
    parser.add_argument("video", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--anchor-frame", type=int, default=400)
    parser.add_argument("--num-frames", type=int, default=120)
    args = parser.parse_args()

    if not args.video.is_file():
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 1

    prepare_mobile_sam_for_run()
    segmenter = get_mobile_sam_segmenter()
    if segmenter is None:
        print("MobileSAM failed to load — check weights and timm.", file=sys.stderr)
        return 1

    detector = PersonDetector()
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Could not open: {args.video}", file=sys.stderr)
        return 1

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    start = max(0, args.anchor_frame)
    end = start + max(1, args.num_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, frame = cap.read()
    if not ok:
        print(f"Could not read anchor frame {start}", file=sys.stderr)
        return 1

    bbox = _largest_bbox(detector(frame))
    if bbox is None:
        print(f"No person detected on anchor frame {start}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        print(f"Could not write: {args.output}", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    drawn = 0
    frame_idx = start
    try:
        while frame_idx < end:
            ok, frame = cap.read()
            if not ok:
                break
            seg = run_segmentation_for_target(
                frame,
                bbox,
                track_id=1,
                is_reid_fallback=False,
                width=w,
                height=h,
                segmenter=segmenter,
            )
            out = frame
            if seg.mask is not None:
                out = _composite_mask(out, seg.mask)
                fx, fy = seg.foot_px
                cv2.circle(out, (int(fx), int(fy)), 6, (0, 0, 255), -1)
                cv2.putText(
                    out,
                    str(seg.segment_source),
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                drawn += 1
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), 2)
            writer.write(out)
            frame_idx += 1
    finally:
        writer.release()
        cap.release()

    print(
        f"Wrote {args.output} frames {start}-{frame_idx - 1} "
        f"({drawn} with SAM masks, anchor bbox on frame {start})"
    )
    return 0 if drawn > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
