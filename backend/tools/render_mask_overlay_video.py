#!/usr/bin/env python3
"""Burn segmentation masks onto a video (CLI smoke test for MobileSAM overlay).

Examples (from repo root, venv active):

  python -m backend.tools.render_mask_overlay_video \\
    path/to/match.mp4 backend/sample-details.json -o /tmp/masked.mp4

  # Reuse rows from a saved analyze JSON (must include mask_rle entries):
  python -m backend.tools.render_mask_overlay_video \\
    path/to/match.mp4 backend/sample-details.json -o /tmp/masked.mp4 \\
    --analyze-json /tmp/analyze.json

If you always get "no mask rows", run without --analyze-json so the pipeline
runs with include_masks_in_response=True (masks kept even when API would strip them).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from backend.app.pipeline.mask_rle import decode_mask_rle
from backend.app.pipeline.run import run_pipeline
from backend.app.schemas import AnalyzeResponse, PlayerDetails, Row


def _rows_from_analyze_json(path: Path) -> tuple[list[Row], float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "rows" not in data:
        raise ValueError("JSON must contain top-level 'rows' (AnalyzeResponse shape).")
    fps = float(data.get("video", {}).get("fps") or 30.0)
    rows = [Row.model_validate(r) for r in data["rows"]]
    return rows, fps


def _composite_mask(
    frame_bgr: np.ndarray,
    mask_hw: np.ndarray,
    *,
    bgr: tuple[int, int, int] = (0, 200, 120),
    alpha: float = 0.45,
) -> np.ndarray:
    """Resize mask to frame if needed; alpha-blend color onto masked pixels."""
    fh, fw = frame_bgr.shape[:2]
    mh, mw = mask_hw.shape
    m = mask_hw
    if (mh, mw) != (fh, fw):
        m = cv2.resize(m.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_NEAREST).astype(
            bool
        )
    if not np.any(m):
        return frame_bgr
    out = frame_bgr.astype(np.float32)
    color = np.array(bgr, dtype=np.float32).reshape(1, 1, 3)
    mm = m[..., np.newaxis].astype(np.float32)
    blended = out * (1.0 - alpha * mm) + color * (alpha * mm)
    return blended.clip(0, 255).astype(np.uint8)


def render(
    video_path: Path,
    rows: list[Row],
    fps: float,
    out_path: Path,
    *,
    max_frames: int | None,
) -> int:
    by_frame: dict[int, Row] = {}
    for r in rows:
        if r.mask_rle is not None:
            by_frame[r.frame] = r

    if not by_frame:
        print(
            "No rows with mask_rle — nothing to draw. "
            "Check SAM weights / timm / lock; or run without --analyze-json.",
            file=sys.stderr,
        )
        return 2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}", file=sys.stderr)
        return 1

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps)
    out_fps = fps if fps > 0 else src_fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (w, h))
    if not writer.isOpened():
        print(f"Could not open VideoWriter for {out_path}", file=sys.stderr)
        cap.release()
        return 1

    frame_idx = 0
    drawn = 0
    try:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            row = by_frame.get(frame_idx)
            if row and row.mask_rle is not None:
                mask = decode_mask_rle(row.mask_rle)
                frame = _composite_mask(frame, mask)
                drawn += 1
            writer.write(frame)
            frame_idx += 1
    finally:
        writer.release()
        cap.release()

    print(
        f"Wrote {out_path} ({frame_idx} frames, green overlay on {drawn} frames with masks)."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render mask overlay video (CLI test for segmentation)."
    )
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("details", type=Path, help="PlayerDetails JSON (same as backend CLI)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output MP4 path",
    )
    parser.add_argument(
        "--analyze-json",
        type=Path,
        default=None,
        help="Use rows from saved analyze JSON instead of re-running pipeline",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap frames written (default: full input read until EOF or MAX_FRAMES in pipeline)",
    )
    args = parser.parse_args()

    if not args.video.is_file():
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 1
    if not args.details.is_file():
        print(f"Details JSON not found: {args.details}", file=sys.stderr)
        return 1

    if args.analyze_json is not None:
        if not args.analyze_json.is_file():
            print(f"Analyze JSON not found: {args.analyze_json}", file=sys.stderr)
            return 1
        rows, fps = _rows_from_analyze_json(args.analyze_json)
    else:
        details = PlayerDetails.model_validate(
            json.loads(args.details.read_text(encoding="utf-8"))
        )
        result = run_pipeline(
            str(args.video),
            details,
            include_masks_in_response=True,
        )
        rows = list(result.rows)
        fps = float(result.video.fps)
        n_mask = sum(1 for r in rows if r.mask_rle is not None)
        cap = os.environ.get("MAX_FRAMES", "5000")
        print(
            f"[render_mask_overlay_video] "
            f"target.track_id={result.target.track_id!r} "
            f"target.method={result.target.method!r} "
            f"heatmap_source={result.heatmap_source!r} "
            f"rows={len(rows)} with_mask_rle={n_mask} "
            f"(MAX_FRAMES={cap})",
            file=sys.stderr,
        )
        if n_mask == 0:
            print(
                "No mask_rle: MobileSAM only runs after a successful player lock + 3-frame warmup. "
                "Common causes: (1) jerseyNumber in your JSON does not match any readable back "
                f"in the first ~{cap} frames (sample-details.json defaults to 10); "
                "(2) lock only happens at end-of-video via color fallback — SAM never runs during the loop; "
                "(3) SAM failed every frame — check SAM_WEIGHTS and timm.",
                file=sys.stderr,
            )

    return render(
        args.video,
        rows,
        fps,
        args.output,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    raise SystemExit(main())
