#!/usr/bin/env python3
"""Generate a sample player heatmap PNG (fast, no full analyze)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from backend.app.pipeline.heatmap import build_heatmap
from backend.app.pipeline.pitch_homography import default_calibration_dir, load_calibration


def _rows_from_detections(
    detections_path: Path,
    *,
    frame_start: int,
    frame_end: int,
    track_id: int | None,
) -> tuple[list[dict], int, float]:
    data = json.loads(detections_path.read_text(encoding="utf-8"))
    meta = data.get("meta", {})
    fps = float(meta.get("fps") or 30.0)
    counts: Counter[int] = Counter()
    by_track: dict[int, list[dict]] = {}

    for fr in data.get("frames", []):
        frame = int(fr.get("frame", 0))
        if frame < frame_start or frame > frame_end:
            continue
        for person in fr.get("people", []):
            tid = int(person["id"])
            counts[tid] += 1
            by_track.setdefault(tid, []).append(
                {
                    "frame": frame,
                    "t": float(fr.get("timestamp", frame / fps)),
                    "track": tid,
                    "bbox": [float(v) for v in person["bbox"]],
                }
            )

    if track_id is None:
        if not counts:
            raise ValueError("No detections in frame range.")
        track_id = counts.most_common(1)[0][0]

    return by_track[track_id], track_id, fps


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo heatmap from detections JSON")
    parser.add_argument(
        "--detections",
        type=Path,
        required=True,
        help="Path to detections JSON (e.g. from a prior analyze export)",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default="testmatch2",
        help="Calibration name under backend/data/pitch_calibration/",
    )
    parser.add_argument("--frame-start", type=int, default=100)
    parser.add_argument("--frame-end", type=int, default=1200)
    parser.add_argument("--track-id", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO / "backend/data/pitch_calibration/demo_heatmap.png",
    )
    args = parser.parse_args()

    cal_path = default_calibration_dir() / f"{args.calibration}.json"
    if not cal_path.is_file():
        raise SystemExit(f"Calibration not found: {cal_path}")

    rows, track_id, fps = _rows_from_detections(
        args.detections,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        track_id=args.track_id,
    )
    cal = load_calibration(cal_path)
    result, image = build_heatmap(rows, cal, fps=fps)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), image)
    print(f"Track {track_id}: {result.sample_count} pitch positions")
    print(f"Saved heatmap: {args.output.resolve()}")


if __name__ == "__main__":
    main()
