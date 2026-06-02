#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2

from ball_detect import BallDetector
from bbox_utils import is_valid_person_bbox
from detect import PersonDetector
from extract_frames import extract_frames
from tracker import PersonTracker, TrackedFrame
from weights_config import resolve_yolo_weights


def _default_weights() -> str:
    return resolve_yolo_weights(None)


def _build_summary(tracked_frames: list[TrackedFrame]) -> dict:
    total_detections = sum(len(f.people) for f in tracked_frames)
    unique_ids = {p.id for f in tracked_frames for p in f.people}
    frame_count = len(tracked_frames)
    avg = total_detections / frame_count if frame_count > 0 else 0.0
    return {
        "total_detections": total_detections,
        "unique_ids": len(unique_ids),
        "avg_people_per_frame": round(avg, 2),
    }


def run_detect(
    video_path: Path,
    output_root: Path,
    weights: str,
    conf: float,
    iou: float,
    track_buffer: int,
    track_high_thresh: float,
    track_low_thresh: float,
    new_track_thresh: float,
    match_thresh: float,
    detect_ball: bool,
    ball_weights: str | None,
    ball_conf: float,
    max_frames: int | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    quiet: bool = False,
) -> tuple[Path, dict[str, Any]]:
    stem = video_path.stem
    run_dir = output_root / stem
    frames_dir = run_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)

    def _log(msg: str) -> None:
        if not quiet:
            print(msg)

    _log(f"Extracting frames from {video_path} ...")
    _, meta = extract_frames(video_path, frames_dir, max_frames=max_frames)
    _log(f"  {meta.total_frames} frames saved to {frames_dir}")

    _log(
        f"Running YOLO + ByteTrack (weights={weights}, conf={conf}, "
        f"track_buffer={track_buffer}) ..."
    )
    detector = PersonDetector(weights=weights, conf=conf, iou=iou)
    tracker = PersonTracker(
        track_buffer=track_buffer,
        track_high_thresh=track_high_thresh,
        track_low_thresh=track_low_thresh,
        new_track_thresh=new_track_thresh,
        match_thresh=match_thresh,
    )

    ball_detector: BallDetector | None = None
    if detect_ball:
        ball_detector = BallDetector.try_create(weights=ball_weights, conf=ball_conf)
        if ball_detector is None:
            _log("Warning: --detect-ball set but ball weights not found; skipping ball.")
        else:
            _log(f"  ball detector: {ball_detector.weights}")

    tracked: list[TrackedFrame] = []
    frame_limit = meta.total_frames if max_frames is None else min(meta.total_frames, max_frames)
    frame_paths = [frames_dir / f"frame_{idx:06d}.jpg" for idx in range(frame_limit)]
    fps = meta.fps if meta.fps > 0 else 30.0

    frames_out = []
    for i, frame_path in enumerate(frame_paths):
        frame_idx = int(frame_path.stem.split("_")[1])
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise SystemExit(f"Could not read frame: {frame_path}")

        detections = detector.detect_frame(frame)
        people = [
            p
            for p in tracker.update(detections, frame)
            if is_valid_person_bbox(p.bbox, frame.shape)
        ]
        tf = TrackedFrame(
            frame=frame_idx,
            timestamp=frame_idx / fps,
            people=people,
        )
        tracked.append(tf)

        frame_entry = {
            "frame": tf.frame,
            "timestamp": round(tf.timestamp, 4),
            "people": [
                {
                    "id": p.id,
                    "bbox": [round(v, 2) for v in p.bbox],
                    "conf": round(p.conf, 4),
                }
                for p in tf.people
            ],
        }
        if ball_detector is not None:
            balls = ball_detector.detect_frame(frame)
            frame_entry["balls"] = [
                {"bbox": [round(v, 2) for v in b.bbox], "conf": round(b.conf, 4)}
                for b in balls
            ]
        frames_out.append(frame_entry)

        if on_progress is not None:
            on_progress(i + 1, len(frame_paths), "detect")
        elif not quiet and (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(frame_paths)} frames")

    _log(f"  processed {len(tracked)} frames")

    config = {
        "weights": weights,
        "conf": conf,
        "iou": iou,
        "tracker": "bytetrack",
        "track_buffer": track_buffer,
        "track_high_thresh": track_high_thresh,
        "track_low_thresh": track_low_thresh,
        "new_track_thresh": new_track_thresh,
        "match_thresh": match_thresh,
        "detect_ball": detect_ball and ball_detector is not None,
    }

    payload = {
        "video": str(video_path.resolve()),
        "meta": {
            "fps": meta.fps,
            "width": meta.width,
            "height": meta.height,
            "total_frames": meta.total_frames,
        },
        "config": config,
        "frames": frames_out,
        "summary": _build_summary(tracked),
    }

    out_json = run_dir / "detections.json"
    out_json.write_text(json.dumps(payload, indent=2))
    _log(f"Wrote {out_json}")
    _log(f"Summary: {payload['summary']}")
    return out_json, payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frames, run YOLOv8 person detection, ByteTrack tracking."
    )
    parser.add_argument("video", type=Path, help="Path to input video")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Root output directory (default: output)",
    )
    parser.add_argument("--weights", default=_default_weights(), help="YOLO weights path")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="YOLO confidence (ByteTrack uses low-conf detections; default 0.1)",
    )
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument(
        "--track-buffer",
        type=int,
        default=30,
        help="Frames to keep lost tracks alive (ByteTrack track_buffer)",
    )
    parser.add_argument(
        "--track-high-thresh",
        type=float,
        default=0.25,
        help="First-stage association threshold for high-confidence detections",
    )
    parser.add_argument(
        "--track-low-thresh",
        type=float,
        default=0.1,
        help="Second-stage threshold for low-confidence detections",
    )
    parser.add_argument(
        "--new-track-thresh",
        type=float,
        default=0.15,
        help="Minimum detection score to start a new track",
    )
    parser.add_argument(
        "--match-thresh",
        type=float,
        default=0.8,
        help="IoU threshold for track-detection association",
    )
    parser.add_argument(
        "--detect-ball",
        action="store_true",
        help="Run optional ball detector; adds balls[] per frame",
    )
    parser.add_argument("--ball-weights", default=None, help="Ball YOLO weights path")
    parser.add_argument("--ball-conf", type=float, default=0.25, help="Ball detection confidence")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only the first N frames (useful for quick tests)",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=None,
        help="Deprecated alias for --track-buffer",
    )
    args = parser.parse_args()

    if not args.video.is_file():
        raise SystemExit(f"Video not found: {args.video}")

    track_buffer = args.max_age if args.max_age is not None else args.track_buffer

    run_detect(
        video_path=args.video,
        output_root=args.output_dir,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        track_buffer=track_buffer,
        track_high_thresh=args.track_high_thresh,
        track_low_thresh=args.track_low_thresh,
        new_track_thresh=args.new_track_thresh,
        match_thresh=args.match_thresh,
        detect_ball=args.detect_ball,
        ball_weights=args.ball_weights,
        ball_conf=args.ball_conf,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
