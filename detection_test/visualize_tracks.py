#!/usr/bin/env python3
"""Draw ByteTrack overlays on frames or source video from detections.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _track_color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    return tuple(int(c) for c in rng.integers(64, 256, size=3))


def _load_detections(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_frame = {entry["frame"]: entry for entry in payload["frames"]}
    return payload, [by_frame[k] for k in sorted(by_frame)]


def _resolve_frame_path(frames_dir: Path, frame_idx: int) -> Path:
    return frames_dir / f"frame_{frame_idx:06d}.jpg"


def _clip_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _draw_people(frame: np.ndarray, people: list[dict], colors: dict[int, tuple[int, int, int]]) -> None:
    height, width = frame.shape[:2]
    for person in people:
        track_id = int(person["id"])
        x1, y1, x2, y2 = (int(round(v)) for v in person["bbox"])
        x1, y1, x2, y2 = _clip_bbox(x1, y1, x2, y2, width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        color = colors.setdefault(track_id, _track_color(track_id))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"id={track_id}"
        if "conf" in person:
            label += f" {person['conf']:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _read_video_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read video frame {frame_idx}")
    return frame


def render_overlay_video(
    payload: dict,
    ordered_frames: list[dict],
    frames_dir: Path | None,
    video_path: Path | None,
    output_path: Path,
    max_frames: int | None = None,
) -> dict:
    meta = payload.get("meta", {})
    fps = float(meta.get("fps") or 25.0)

    if max_frames is not None:
        ordered_frames = ordered_frames[:max_frames]

    if not ordered_frames:
        raise SystemExit("No frames found in detections.json")

    colors: dict[int, tuple[int, int, int]] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    cap: cv2.VideoCapture | None = None
    out_size: tuple[int, int] | None = None

    if frames_dir is None:
        if video_path is None or not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

    try:
        for entry in ordered_frames:
            frame_idx = int(entry["frame"])
            frame: np.ndarray | None = None
            if frames_dir is not None:
                frame_path = _resolve_frame_path(frames_dir, frame_idx)
                if frame_path.is_file():
                    frame = cv2.imread(str(frame_path))
                    if frame is None:
                        raise RuntimeError(f"Could not read frame image: {frame_path}")
                elif video_path is not None:
                    if cap is None:
                        cap = cv2.VideoCapture(str(video_path))
                        if not cap.isOpened():
                            raise RuntimeError(f"Could not open video: {video_path}")
                    frame = _read_video_frame(cap, frame_idx)
                else:
                    raise FileNotFoundError(f"Missing frame image: {frame_path}")
            else:
                assert cap is not None
                frame = _read_video_frame(cap, frame_idx)

            h, w = frame.shape[:2]
            if out_size is None:
                out_size = (w, h)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, out_size)
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer: {output_path}")

            if (w, h) != out_size:
                frame = cv2.resize(frame, out_size)

            _draw_people(frame, entry.get("people", []), colors)
            cv2.putText(
                frame,
                f"frame={frame_idx}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            assert writer is not None
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()
        if cap is not None:
            cap.release()

    return {
        "frames_rendered": len(ordered_frames),
        "unique_ids": len(colors),
        "output": str(output_path.resolve()),
    }


def id_switch_report(ordered_frames: list[dict]) -> dict:
    """Heuristic: count track-id changes in a sliding bbox neighborhood."""
    switches = 0
    comparisons = 0
    prev_by_cell: dict[tuple[int, int], int] = {}

    for entry in ordered_frames:
        next_by_cell: dict[tuple[int, int], int] = {}
        for person in entry.get("people", []):
            x1, y1, x2, y2 = person["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            cell = (int(cx // 80), int(cy // 80))
            track_id = int(person["id"])
            prev_id = prev_by_cell.get(cell)
            if prev_id is not None and prev_id != track_id:
                switches += 1
            comparisons += 1
            next_by_cell[cell] = track_id
        prev_by_cell = next_by_cell

    return {
        "grid_cell_px": 80,
        "approx_id_switches": switches,
        "cell_observations": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ByteTrack detections.json overlays.")
    parser.add_argument("detections", type=Path, help="Path to detections.json")
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Directory with frame_*.jpg (default: sibling frames/ next to detections.json)",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Source video if frames/ is missing or incomplete",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output mp4 path (default: <detections_dir>/tracked_overlay.mp4)",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limit rendered frames")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print approximate ID-switch report to stdout",
    )
    args = parser.parse_args()

    if not args.detections.is_file():
        raise SystemExit(f"Detections not found: {args.detections}")

    payload, ordered_frames = _load_detections(args.detections)
    run_dir = args.detections.parent
    frames_dir = args.frames_dir
    if frames_dir is None and (run_dir / "frames").is_dir():
        frames_dir = run_dir / "frames"

    video_path = args.video
    if video_path is None:
        video_raw = payload.get("video")
        if video_raw and Path(video_raw).is_file():
            video_path = Path(video_raw)

    if frames_dir is None or not frames_dir.is_dir():
        if video_path is None:
            raise SystemExit(
                f"Frames dir not found ({run_dir / 'frames'}). Pass --video or ensure frames/ exists."
            )
        frames_dir = None

    output_path = args.output or (run_dir / "tracked_overlay.mp4")

    if args.report:
        frames_for_report = ordered_frames
        if args.max_frames is not None:
            frames_for_report = ordered_frames[: args.max_frames]
        print(json.dumps(id_switch_report(frames_for_report), indent=2))

    summary = render_overlay_video(
        payload=payload,
        ordered_frames=ordered_frames,
        frames_dir=frames_dir,
        video_path=video_path,
        output_path=output_path,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
