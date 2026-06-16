#!/usr/bin/env python3
"""Run lock-acquisition root-cause audit on a video clip."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.pipeline.run import run_pipeline
from backend.app.schemas import PlayerDetails


def main() -> None:
    video = ROOT / "testing video" / "testmatch2.mp4"
    if not video.is_file():
        raise SystemExit(f"Missing video: {video}")

    frame_cap = int(os.environ.get("AUDIT_FRAME_CAP", "4000"))
    out_dir = ROOT / "backend" / "data" / "lock_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug_ocr"
    report_path = out_dir / f"testmatch2_audit_{frame_cap}f.json"

    os.environ["LOCK_AUDIT"] = "1"
    os.environ["LOCK_AUDIT_DIR"] = str(debug_dir)
    os.environ["LOCK_AUDIT_REPORT"] = str(report_path)
    os.environ["TARGET_MEMORY_ENABLED"] = "0"
    os.environ["MAX_FRAMES"] = str(frame_cap)
    os.environ["SAM_ENABLED"] = "1"
    os.environ["SAM_STRIDE"] = os.environ.get("AUDIT_SAM_STRIDE", "3")
    os.environ["TRACKER_TYPE"] = "botsort"
    os.environ["TRACK_BUFFER"] = "120"
    os.environ["TRACK_MIN_AGE"] = "3"
    os.environ["OCR_EVERY"] = "5"

    details = PlayerDetails(
        sport="football",
        name="H. Lozano",
        jerseyNumber=22,
        primaryJerseyColor="#ffffff",
        secondaryJerseyColor="#000000",
        teamName="Home",
    )

    print(f"=== Lock audit: {video.name} ({frame_cap} frames) ===", flush=True)
    t0 = time.time()
    resp = run_pipeline(
        str(video),
        details,
        calibration_key="testmatch2",
        upload_filename=video.name,
    )
    elapsed = time.time() - t0

    report = json.loads(report_path.read_text())
    summary = {
        "video": str(video),
        "frames_processed": resp.video.frames,
        "elapsed_s": round(elapsed, 2),
        "target": {
            "method": resp.target.method,
            "track_id": resp.target.track_id,
            "locked_at_frame": resp.target.locked_at_frame,
        },
        "audit": report,
        "diagnosis": _diagnose(report),
    }
    diagnosis_path = out_dir / f"testmatch2_diagnosis_{frame_cap}f.json"
    diagnosis_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"WROTE {report_path}", flush=True)
    print(f"WROTE {diagnosis_path}", flush=True)


def _diagnose(report: dict) -> dict:
    jersey = report.get("target_jersey", 22)
    reads_key = f"ocr_reads_{jersey}"
    ocr_reads_target = report.get(reads_key, 0)
    locks_created = report.get("locks_created", 0)
    lock_resets = report.get("lock_resets", 0)
    match_successes = report.get("successful_matches", 0)

    if ocr_reads_target == 0:
        outcome = "A"
        conclusion = f"OCR never saw jersey {jersey}."
        focus = "OCR quality, crop quality, OCR frequency"
    elif ocr_reads_target > 0 and locks_created == 0:
        if match_successes > 0:
            outcome = "D"
            conclusion = "Match successes recorded but lock creation failed."
            focus = "identity matcher, lock state transition, candidate promotion"
        else:
            outcome = "B"
            conclusion = "OCR read target jersey but matching logic rejected all reads."
            focus = "thresholds, filters, validation logic"
    elif locks_created > 0 and lock_resets > 0:
        outcome = "C"
        conclusion = "Lock formed then destroyed."
        focus = "scene cuts, tracker reset, confidence decay"
    elif locks_created > 0:
        outcome = "OK"
        conclusion = "Lock acquired successfully."
        focus = ""
    else:
        outcome = "UNKNOWN"
        conclusion = "Insufficient evidence."
        focus = "review lock_audit.log and debug_ocr images"

    return {
        "outcome": outcome,
        "conclusion": conclusion,
        "focus": focus,
    }


if __name__ == "__main__":
    main()
