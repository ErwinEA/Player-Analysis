#!/usr/bin/env python3
"""Sandbox A/B: baseline lock vs TargetMemory-enabled pipeline."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.pipeline.run import run_pipeline
from backend.app.schemas import PlayerDetails


def _apply_env(env: dict[str, str]) -> None:
    keys = [
        "TARGET_MEMORY_ENABLED",
        "TARGET_MEMORY_DEBUG",
        "MAX_FRAMES",
        "SAM_STRIDE",
        "SAM_ENABLED",
        "TRACKER_TYPE",
        "TRACK_BUFFER",
        "TRACK_MIN_AGE",
        "OCR_EVERY",
    ]
    for key in keys:
        if key in env:
            os.environ[key] = env[key]
        elif key in os.environ:
            del os.environ[key]


def _summarize(resp, run_id: str, elapsed_s: float) -> dict:
    rows = resp.rows or []
    vf = int(resp.video.frames)
    tgt = resp.target
    target_rows = [r for r in rows if r.player is not None]
    tframes = sorted({r.frame for r in target_rows})
    tset = set(tframes)
    gaps = [
        tframes[i] - tframes[i - 1] - 1
        for i in range(1, len(tframes))
        if tframes[i] - tframes[i - 1] > 1
    ]
    locked_at = tgt.locked_at_frame
    post_lock_cov = 0.0
    if locked_at is not None and vf > locked_at:
        seen = sum(1 for f in range(locked_at, vf) if f in tset)
        post_lock_cov = seen / (vf - locked_at)

    seg = Counter((r.segment_source or "none") for r in target_rows)
    tracks = [r.track for r in target_rows]

    return {
        "run_id": run_id,
        "elapsed_s": round(elapsed_s, 2),
        "target": {
            "method": tgt.method,
            "track_id": tgt.track_id,
            "confidence": tgt.confidence,
            "locked_at_frame": locked_at,
            "segmentation_started_at_frame": tgt.segmentation_started_at_frame,
        },
        "lock_quality": {
            "has_lock": tgt.track_id is not None,
            "lock_none_frames": vf - len(tset) if tgt.track_id else vf,
            "target_frame_coverage": round(len(tset) / vf, 4) if vf else 0.0,
            "post_lock_coverage": round(post_lock_cov, 4),
            "time_to_lock_s": round(locked_at / resp.video.fps, 2)
            if locked_at is not None and resp.video.fps
            else None,
        },
        "continuity": {
            "target_rows": len(target_rows),
            "unique_target_frames": len(tset),
            "longest_gap_frames": max(gaps) if gaps else 0,
            "gap_events": len(gaps),
            "unique_target_tracks": sorted(set(tracks)),
            "target_track_switches": sum(
                1 for i in range(1, len(tracks)) if tracks[i] != tracks[i - 1]
            ),
        },
        "segmentation": {
            "masks_available": resp.masks_available,
            "mask_row_count": resp.mask_row_count,
            "segment_source_counts": dict(seg),
        },
        "movement_distance_m": resp.movement.distance_m if resp.movement else None,
        "heatmap_samples": resp.heatmap.sample_count if resp.heatmap else 0,
    }


def main() -> None:
    video = ROOT / "testing video" / "testmatch2.mp4"
    if not video.is_file():
        raise SystemExit(f"Missing video: {video}")

    frame_cap = int(os.environ.get("AB_FRAME_CAP", "4000"))
    out_dir = ROOT / "backend" / "data" / "ab_sandbox"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"testmatch2_ab_{frame_cap}f.json"

    details = PlayerDetails(
        sport="football",
        name="Player 22",
        jerseyNumber=22,
        primaryJerseyColor="#ffffff",
        secondaryJerseyColor="#000000",
        teamName="Home",
    )

    shared = {
        "MAX_FRAMES": str(frame_cap),
        "SAM_ENABLED": "1",
        "SAM_STRIDE": os.environ.get("AB_SAM_STRIDE", "3"),
        "TRACKER_TYPE": "botsort",
        "TRACK_BUFFER": "120",
        "TRACK_MIN_AGE": "3",
        "OCR_EVERY": "5",
    }

    runs = []
    for run_id, memory_flag in (("A_baseline", "0"), ("B_target_memory", "1")):
        env = {
            **shared,
            "TARGET_MEMORY_ENABLED": memory_flag,
            "TARGET_MEMORY_DEBUG": os.environ.get("TARGET_MEMORY_DEBUG", "0"),
        }
        _apply_env(env)
        print(f"=== {run_id} TARGET_MEMORY_ENABLED={memory_flag} ===", flush=True)
        t0 = time.time()
        resp = run_pipeline(
            str(video),
            details,
            calibration_key="testmatch2",
            upload_filename=video.name,
        )
        elapsed = time.time() - t0
        rec = _summarize(resp, run_id, elapsed)
        rec["env"] = env
        runs.append(rec)
        print(json.dumps(rec, indent=2), flush=True)

    a, b = runs[0], runs[1]
    comparison = {
        "frame_cap": frame_cap,
        "has_lock_A": a["lock_quality"]["has_lock"],
        "has_lock_B": b["lock_quality"]["has_lock"],
        "method_A": a["target"]["method"],
        "method_B": b["target"]["method"],
        "locked_at_A": a["target"]["locked_at_frame"],
        "locked_at_B": b["target"]["locked_at_frame"],
        "time_to_lock_s_A": a["lock_quality"]["time_to_lock_s"],
        "time_to_lock_s_B": b["lock_quality"]["time_to_lock_s"],
        "target_coverage_A": a["lock_quality"]["target_frame_coverage"],
        "target_coverage_B": b["lock_quality"]["target_frame_coverage"],
        "post_lock_coverage_A": a["lock_quality"]["post_lock_coverage"],
        "post_lock_coverage_B": b["lock_quality"]["post_lock_coverage"],
        "longest_gap_A": a["continuity"]["longest_gap_frames"],
        "longest_gap_B": b["continuity"]["longest_gap_frames"],
        "track_switches_A": a["continuity"]["target_track_switches"],
        "track_switches_B": b["continuity"]["target_track_switches"],
        "mask_rows_A": a["segmentation"]["mask_row_count"],
        "mask_rows_B": b["segmentation"]["mask_row_count"],
        "movement_m_A": a["movement_distance_m"],
        "movement_m_B": b["movement_distance_m"],
    }

    payload = {
        "video": str(video),
        "details": details.model_dump(),
        "runs": runs,
        "comparison": comparison,
        "generated_at_unix": time.time(),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"WROTE {out_path}", flush=True)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
