from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoMeta:
    fps: float
    width: int
    height: int
    total_frames: int


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    max_frames: int | None = None,
) -> tuple[Path, VideoMeta]:
    """Extract every frame from a video to output_dir/frame_{idx:06d}.jpg."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        reported_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        frame_idx = 0
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            out_path = output_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            frame_idx += 1

        total_frames = frame_idx if frame_idx > 0 else reported_total
        meta = VideoMeta(fps=fps, width=width, height=height, total_frames=total_frames)
        return output_dir, meta
    finally:
        cap.release()
