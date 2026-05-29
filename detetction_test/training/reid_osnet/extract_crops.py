#!/usr/bin/env python3
"""Extract player crops from SoccerNet tracking annotations + videos.

For ReID-only (no videos), use prepare_reid_crops.py instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from common import load_config

TRACKING_JSON_NAMES = (
    "tracking_data.json",
    "tracking.json",
    "labels.json",
)


def _iter_tracking_files(data_root: Path) -> list[Path]:
    found: list[Path] = []
    for name in TRACKING_JSON_NAMES:
        found.extend(data_root.rglob(name))
    # Also search inside unpacked tracking/ zips
    tracking_dir = data_root / "tracking"
    if tracking_dir.is_dir():
        for name in TRACKING_JSON_NAMES:
            found.extend(tracking_dir.rglob(name))
    return sorted(set(found))


def _video_for_annotation(ann_path: Path, data_root: Path) -> Path | None:
    """Resolve video near annotation or under standard SoccerNet game folders."""
    parent = ann_path.parent
    for pattern in ("*.mp4", "*.mkv", "*.avi"):
        matches = sorted(parent.glob(pattern))
        if matches:
            return matches[0]
    for pattern in ("*.mp4", "*.mkv", "*.avi"):
        matches = sorted(parent.parent.glob(pattern))
        if matches:
            return matches[0]

    # SoccerNet v2 game path: league/season/game/1_720p.mkv
    parts = ann_path.parts
    for i in range(len(parts) - 1, -1, -1):
        if " - " in parts[i] and i >= 2:
            game_dir = Path(*parts[: i + 1])
            if game_dir.is_dir():
                for pattern in ("1_720p.mkv", "2_720p.mkv", "1.mkv", "2.mkv", "*.mp4"):
                    matches = sorted(game_dir.glob(pattern))
                    if matches:
                        return matches[0]
            break
    return None


def _parse_entries(tracking: object) -> list[dict]:
    if isinstance(tracking, list):
        return [e for e in tracking if isinstance(e, dict)]
    if isinstance(tracking, dict):
        for key in ("data", "annotations", "tracks", "frames"):
            if key in tracking and isinstance(tracking[key], list):
                return [e for e in tracking[key] if isinstance(e, dict)]
    return []


def extract_crops(
    data_root: Path,
    crops_dir: Path,
    min_height: int = 50,
    max_videos: int | None = None,
) -> dict:
    crops_dir.mkdir(parents=True, exist_ok=True)
    ann_files = _iter_tracking_files(data_root)
    if not ann_files:
        print(
            f"No tracking JSON found under {data_root}.\n"
            "Options:\n"
            "  1) ReID (no video): python prepare_reid_crops.py\n"
            "  2) Tracking zip: download_soccernet.py --tasks tracking --unpack\n"
            "  3) Videos: download_soccernet.py --videos 720p --video-split train",
            file=sys.stderr,
        )
        return {"videos": 0, "crops": 0, "players": 0}

    if max_videos:
        ann_files = ann_files[:max_videos]

    total_crops = 0
    player_ids: set[str] = set()
    videos_ok = 0

    for ann_path in ann_files:
        video_path = _video_for_annotation(ann_path, data_root)
        if video_path is None or not video_path.is_file():
            print(f"  skip (no video): {ann_path}")
            continue

        try:
            tracking = json.loads(ann_path.read_text())
        except json.JSONDecodeError as exc:
            print(f"  skip (bad json): {ann_path} ({exc})")
            continue

        entries = _parse_entries(tracking)
        if not entries:
            print(f"  skip (no entries): {ann_path}")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  skip (cannot open): {video_path}")
            continue

        camera_id = ann_path.parent.name
        videos_ok += 1
        frame_cache: dict[int, object] = {}

        for entry in entries:
            frame_idx = int(entry.get("frame", entry.get("frame_idx", entry.get("frame_index", -1))))
            player_id = str(
                entry.get("player_id", entry.get("obj_id", entry.get("track_id", "unknown")))
            )
            jersey = str(entry.get("jersey_number", entry.get("jersey", "0")))
            bbox = entry.get("bbox") or entry.get("bbox_ltwh") or entry.get("box")
            if frame_idx < 0 or bbox is None:
                continue

            if len(bbox) == 4:
                if float(bbox[2]) < float(bbox[0]) or float(bbox[3]) < float(bbox[1]):
                    x1, y1, w, h = map(float, bbox)
                    x2, y2 = x1 + w, y1 + h
                else:
                    x1, y1, x2, y2 = map(float, bbox)
            else:
                continue

            if (y2 - y1) < min_height:
                continue

            if frame_idx not in frame_cache:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                frame_cache[frame_idx] = frame
            else:
                frame = frame_cache[frame_idx]

            ih, iw = frame.shape[:2]
            x1i = max(0, int(x1))
            y1i = max(0, int(y1))
            x2i = min(iw, int(x2))
            y2i = min(ih, int(y2))
            if x2i <= x1i or y2i <= y1i:
                continue

            crop = frame[y1i:y2i, x1i:x2i]
            out_dir = crops_dir / player_id / jersey
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{camera_id}_f{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), crop)
            total_crops += 1
            player_ids.add(player_id)

        cap.release()
        print(f"  {video_path.name}: crops so far {total_crops}")

    summary = {"videos": videos_ok, "crops": total_crops, "players": len(player_ids)}
    (crops_dir / "extract_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SoccerNet player crops from tracking + video.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--crops-dir", type=Path, default=None)
    parser.add_argument("--min-height", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = Path(args.data_root or cfg["data_root"])
    crops_dir = Path(args.crops_dir or cfg["crops_dir"])
    min_height = args.min_height if args.min_height is not None else cfg["min_crop_height"]

    print(f"Extracting crops from {data_root} -> {crops_dir} (min_height={min_height})")
    summary = extract_crops(data_root, crops_dir, min_height, args.max_videos)
    print(f"Done: {summary}")


if __name__ == "__main__":
    main()
