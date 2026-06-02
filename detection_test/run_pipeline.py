#!/usr/bin/env python3
"""End-to-end pipeline: YOLOv8 → filter → ByteTrack → jersey labels → JSON (+ optional target lock)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cli_ui import PipelineUI, StageTimer
from identify import IdentifyConfig, TargetProfile
from identify_runner import run_identify
from jersey_number import JerseyProfileReader
from label_tracks import LabelTracksConfig, label_detections
from movement_stats import compute_movement_stats_from_rows
from run_detect import run_detect
from weights_config import resolve_yolo_weights


SCHEMA_VERSION = 1


def _build_identify_config(args: argparse.Namespace) -> IdentifyConfig:
    face_weight = 0.0 if args.photo is None else args.face_weight
    return IdentifyConfig(
        face_weight=face_weight,
        kit_hist_gate=args.kit_hist_gate,
        min_evidence_reads=args.min_evidence_reads,
        evidence_window_frames=args.evidence_window_frames,
        min_read_spacing=args.min_read_spacing,
        high_conf_classifier=args.high_conf_classifier,
        high_conf_ocr=args.high_conf_ocr,
        target_number_min_conf=args.target_number_min_conf,
        kit_pixel_gate=args.kit_pixel_gate,
        kit_pixel_tolerance=args.kit_pixel_tolerance,
        skip_kit_gate=args.no_kit_gate,
        require_confirmed_for_lock=not args.allow_face_only_lock,
        ocr_every=args.jersey_every,
        relock_max=args.relock_max,
    )


def run_pipeline(args: argparse.Namespace) -> Path:
    ui = PipelineUI(total_steps=4 if args.target_number is not None else 3)
    ui.banner()

    if not args.video.is_file():
        raise SystemExit(f"Video not found: {args.video}")

    run_dir = args.output_dir.resolve() / args.video.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: YOLO + ByteTrack + geometry filter ─────────────────────────
    ui.step_header(1, "Detect & track players", "YOLOv8 person detection + ByteTrack IDs")
    ui.info(f"Video: {args.video.name}")
    ui.info(f"Output folder: {run_dir}")
    timer = StageTimer()

    def on_detect_progress(current: int, total: int, _stage: str) -> None:
        ui.progress(current, total, prefix="Frames")

    detections_path, detect_payload = run_detect(
        video_path=args.video,
        output_root=args.output_dir,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        track_buffer=args.track_buffer,
        track_high_thresh=args.track_high_thresh,
        track_low_thresh=args.track_low_thresh,
        new_track_thresh=args.new_track_thresh,
        match_thresh=args.match_thresh,
        detect_ball=args.detect_ball,
        ball_weights=args.ball_weights,
        ball_conf=args.ball_conf,
        max_frames=args.max_frames,
        on_progress=on_detect_progress,
        quiet=True,
    )
    ui.stage_timing("Detect + track", timer.elapsed())
    summary = detect_payload.get("summary", {})
    ui.success(
        f"{summary.get('unique_ids', '?')} track IDs, "
        f"{summary.get('total_detections', '?')} detections"
    )

    # ── Step 2: Jersey labeling ─────────────────────────────────────────────
    ui.step_header(2, "Jersey numbers", "Gatekeeper + EfficientNet-B0 (per track)")
    timer = StageTimer()
    reader = JerseyProfileReader()
    label_cfg = LabelTracksConfig(
        jersey_every=args.jersey_every,
        kit_hist_gate=args.kit_hist_gate,
        kit_pixel_gate=args.kit_pixel_gate,
        kit_pixel_tolerance=args.kit_pixel_tolerance,
        use_kit_gate=not args.no_kit_gate,
        primary_color=args.primary_color,
        secondary_color=args.secondary_color,
    )

    def on_label_progress(current: int, total: int) -> None:
        ui.progress(current, total, prefix="Label")

    label_detections(
        detect_payload,
        detections_path.parent / "frames",
        reader,
        label_cfg,
        on_progress=on_label_progress,
        max_frames=args.max_frames,
    )
    ui.stage_timing("Jersey labeling", timer.elapsed())
    lstats = detect_payload.get("label_stats", {})
    ui.success(
        f"{lstats.get('jersey_reads', 0)} reads, "
        f"{lstats.get('labeled_detections', 0)} with a number"
    )

    labeled_path = run_dir / "detections_labeled.json"
    labeled_path.write_text(json.dumps(detect_payload, indent=2))

    # ── Step 3: Optional target player lock ─────────────────────────────────
    target_result: dict[str, Any] | None = None
    step = 3
    if args.target_number is not None:
        ui.step_header(
            step,
            f"Lock target #{args.target_number}",
            f"{args.target_name} — Gatekeeper + evidence buffer",
        )
        timer = StageTimer()
        profile = TargetProfile(
            name=args.target_name,
            number=args.target_number,
            primary_color=args.primary_color,
            secondary_color=args.secondary_color,
        )
        id_config = _build_identify_config(args)

        def on_id_progress(current: int, total: int, _stage: str) -> None:
            ui.progress(current, total, prefix="Identify")

        target_result = run_identify(
            labeled_path,
            config=id_config,
            profile=profile,
            photo=args.photo,
            max_frames=args.max_frames,
            on_progress=on_id_progress,
        )
        ui.stage_timing("Target identification", timer.elapsed())
        isum = target_result.get("summary", {})
        ui.success(
            f"Locked {isum.get('locked_frames', 0)} frames, "
            f"{isum.get('lock_count', 0)} segment(s)"
        )
        identified_path = run_dir / "identified.json"
        identified_path.write_text(json.dumps(target_result, indent=2))
        step += 1

    # ── Step 4: Write unified pipeline.json ─────────────────────────────────
    ui.step_header(step, "Write pipeline.json", "Single file with all stages")
    pipeline_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "video": str(args.video.resolve()),
        "meta": detect_payload.get("meta", {}),
        "stages": {
            "detect": detect_payload.get("config", {}),
            "label": asdict(label_cfg),
        },
        "detect_summary": detect_payload.get("summary", {}),
        "label_stats": detect_payload.get("label_stats", {}),
        "tracks": detect_payload.get("tracks", {}),
        "frames": detect_payload.get("frames", []),
    }
    if target_result is not None:
        pipeline_payload["stages"]["identify"] = target_result.get("config", {})
        pipeline_payload["target"] = {
            "profile": target_result.get("target_profile"),
            "locked_segments": target_result.get("locked_segments", []),
            "rows": target_result.get("rows", []),
            "summary": target_result.get("summary", {}),
            "diagnostics": target_result.get("diagnostics", {}),
        }
        meta = detect_payload.get("meta", {})
        fps = float(meta.get("fps") or 30.0)
        target_rows = target_result.get("rows", [])
        if len(target_rows) >= 2:
            cal_path = (
                Path(__file__).resolve().parents[1]
                / "backend"
                / "data"
                / "pitch_calibration"
                / f"{args.video.stem}.json"
            )
            movement = compute_movement_stats_from_rows(
                target_rows,
                fps=fps,
                calibration_path=cal_path if cal_path.is_file() else None,
            )
            pipeline_payload["movement_stats"] = movement.to_dict()
            sprint_note = (
                f", {movement.sprint_count} sprints"
                if movement.units == "meters"
                else " (sprints need pitch calibration)"
            )
            ui.info(
                f"Movement: {movement.distance_m:.1f} {movement.units}, "
                f"max {movement.max_speed_m_s:.1f}, "
                f"avg {movement.avg_speed_m_s:.1f}{sprint_note}"
            )

    pipeline_path = run_dir / "pipeline.json"
    pipeline_path.write_text(json.dumps(pipeline_payload, indent=2))
    ui.success(f"Saved {pipeline_path.name}")

    # ── Final summary ───────────────────────────────────────────────────────
    track_count = len(detect_payload.get("tracks", {}))
    numbered = sum(
        1 for t in detect_payload.get("tracks", {}).values() if t.get("jersey_number") is not None
    )
    ui.summary_box(
        [
            ("Frames processed", str(len(detect_payload.get("frames", [])))),
            ("Unique tracks", str(summary.get("unique_ids", "?"))),
            ("Tracks with jersey #", str(numbered)),
            ("Pipeline JSON", str(pipeline_path.name)),
            ("Labeled detections", str(labeled_path.name)),
        ]
    )
    ui.next_steps(
        [
            f"Open {pipeline_path} for full results",
            f"python visualize_tracks.py {labeled_path} --report",
            f"Inspect frames in {run_dir / 'frames'}",
        ]
    )
    return pipeline_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-command pipeline: YOLOv8 → filter → ByteTrack → jersey → pipeline.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python run_pipeline.py video.mp4 \\
    --target-number 23 --target-name "Player" \\
    --primary-color "#1F3D31" --secondary-color "#000000"

Quick test (first 300 frames):
  python run_pipeline.py video.mp4 --max-frames 300 --target-number 10 \\
    --primary-color "#ffffff" --secondary-color "#000000" --target-name "Test"
        """.strip(),
    )
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))

    parser.add_argument("--weights", default=None, help="YOLO weights (default: soccer or yolov8m)")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--track-high-thresh", type=float, default=0.25)
    parser.add_argument("--track-low-thresh", type=float, default=0.1)
    parser.add_argument("--new-track-thresh", type=float, default=0.15)
    parser.add_argument("--match-thresh", type=float, default=0.8)
    parser.add_argument("--detect-ball", action="store_true")
    parser.add_argument("--ball-weights", default=None)
    parser.add_argument("--ball-conf", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=None)

    parser.add_argument("--jersey-every", type=int, default=5, help="Frames between jersey reads per track")
    parser.add_argument("--kit-hist-gate", type=float, default=0.35)
    parser.add_argument("--kit-pixel-gate", type=float, default=0.12)
    parser.add_argument("--kit-pixel-tolerance", type=int, default=40)
    parser.add_argument("--no-kit-gate", action="store_true", help="Label all valid bboxes (skip kit color gate)")

    parser.add_argument("--target-number", type=int, default=None, help="Lock this jersey number (optional)")
    parser.add_argument("--target-name", type=str, default="Player")
    parser.add_argument("--primary-color", type=str, default="#1F3D31")
    parser.add_argument("--secondary-color", type=str, default="#000000")
    parser.add_argument("--photo", type=Path, default=None, help="Reference face photo (optional)")
    parser.add_argument("--face-weight", type=float, default=2.0)
    parser.add_argument("--allow-face-only-lock", action="store_true")

    parser.add_argument("--min-evidence-reads", type=int, default=3)
    parser.add_argument("--evidence-window-frames", type=int, default=90)
    parser.add_argument("--min-read-spacing", type=int, default=5)
    parser.add_argument("--high-conf-classifier", type=float, default=0.75)
    parser.add_argument("--high-conf-ocr", type=float, default=0.55)
    parser.add_argument("--target-number-min-conf", type=float, default=0.25)
    parser.add_argument(
        "--relock-max",
        type=int,
        default=0,
        help="Max frames in LOST before give up (0 = entire video)",
    )

    args = parser.parse_args()
    if args.photo is not None and not args.photo.is_file():
        raise SystemExit(f"Photo not found: {args.photo}")
    if args.weights is None:
        args.weights = resolve_yolo_weights(None)

    for label, hex_color in (("primary", args.primary_color), ("secondary", args.secondary_color)):
        try:
            from color_score import _hex_to_rgb

            _hex_to_rgb(hex_color)
        except ValueError as exc:
            raise SystemExit(f"Invalid {label} color: {exc}") from exc

    if args.target_number is None:
        print("Note: --target-number not set; skipping target lock stage (tracks still labeled).\n")

    run_pipeline(args)


if __name__ == "__main__":
    main()
