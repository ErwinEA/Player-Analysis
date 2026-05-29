"""Run target-player identification and return a JSON-serializable result."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

import cv2
import json

from face_match import FaceMatcher, NullFaceMatcher
from identify import IdentifyConfig, IdentifyEngine, TargetProfile
from jersey_number import JerseyProfileReader


def run_identify(
    detections_path: Path,
    *,
    config: IdentifyConfig,
    profile: TargetProfile,
    photo: Path | None = None,
    max_frames: int | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    payload = json.loads(detections_path.read_text())
    frames_dir = detections_path.parent / "frames"
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    meta = payload["meta"]
    fps = float(meta.get("fps") or 30.0)
    total_frames = int(meta.get("total_frames") or len(payload.get("frames", [])))

    run_config = config
    if config.relock_max <= 0:
        run_config = replace(config, relock_max=total_frames or 999999)

    face_matcher = FaceMatcher(photo) if photo is not None else NullFaceMatcher()
    ocr = JerseyProfileReader()
    engine = IdentifyEngine(
        config=run_config, profile=profile, face_matcher=face_matcher, ocr=ocr
    )

    frames_data = sorted(payload.get("frames", []), key=lambda f: f["frame"])
    if max_frames is not None:
        frames_data = frames_data[:max_frames]

    for i, frame_data in enumerate(frames_data):
        frame_idx = int(frame_data["frame"])
        timestamp = float(frame_data.get("timestamp", frame_idx / fps if fps > 0 else 0.0))
        people = frame_data.get("people", [])
        frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
        frame_img = cv2.imread(str(frame_path))
        if frame_img is None:
            raise FileNotFoundError(f"Could not read frame: {frame_path}")
        engine.process_frame(frame_idx, timestamp, frame_img, people)
        if on_progress is not None:
            on_progress(i + 1, len(frames_data), "identify")

    engine.finalize()

    return {
        "source_detections": str(detections_path),
        "target_profile": {
            "name": profile.name,
            "number": profile.number,
            "primary_color": profile.primary_color,
            "secondary_color": profile.secondary_color,
            "confirmed_number": profile.confirmed_number,
            "confirmed_name": profile.confirmed_name,
        },
        "config": asdict(run_config),
        "locked_segments": [
            {
                "start_frame": seg.start_frame,
                "end_frame": seg.end_frame,
                "track_id": seg.track_id,
            }
            for seg in engine.locked_segments
        ],
        "rows": [
            {
                "frame": row.frame,
                "t": row.t,
                "track": row.track,
                "bbox": row.bbox,
                "score": row.score,
                "method": row.method,
            }
            for row in engine.rows
        ],
        "summary": engine.summary(total_frames),
        "diagnostics": {
            "track_evidence": engine.evidence_diagnostics(),
            "confirmed_track_ids": [
                tid for tid, buf in engine.track_evidence.items() if buf.confirmed
            ],
        },
    }
