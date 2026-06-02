#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from identify import IdentifyConfig, TargetProfile
from identify_runner import run_identify


def _build_config(args: argparse.Namespace) -> IdentifyConfig:
    face_weight = 0.0 if args.photo is None else args.face_weight
    return IdentifyConfig(
        face_weight=face_weight,
        num_weight=args.num_weight,
        name_weight=args.name_weight,
        color_weight=args.color_weight,
        lock_threshold=args.lock_threshold,
        margin=args.margin,
        initial_frames=args.initial_frames,
        initial_face_score=args.initial_face_score,
        relock_window=args.relock_window,
        relock_cooldown=args.relock_cooldown,
        relock_max=args.relock_max,
        ocr_every=args.ocr_every,
        ocr_votes=args.ocr_votes,
        ocr_reads=args.ocr_reads,
        name_min_sim=args.name_min_sim,
        color_gate=args.color_gate,
        color_tolerance=args.color_tolerance,
        absence_frames=args.absence_frames,
        number_min_conf=args.number_min_conf,
        zero_face_unlock=args.zero_face_unlock,
        relock_face_score=args.relock_face_score,
        handoff_face_min=args.handoff_face_min,
        handoff_margin=args.handoff_margin,
        handoff_max_frames=args.handoff_max_frames,
        kit_hist_gate=args.kit_hist_gate,
        min_evidence_reads=args.min_evidence_reads,
        evidence_window_frames=args.evidence_window_frames,
        min_read_spacing=args.min_read_spacing,
        high_conf_classifier=args.high_conf_classifier,
        high_conf_ocr=args.high_conf_ocr,
        target_number_min_conf=args.target_number_min_conf,
        kit_pixel_gate=args.kit_pixel_gate,
        kit_pixel_tolerance=args.kit_pixel_tolerance,
        skip_kit_gate=False,
        require_confirmed_for_lock=not args.allow_face_only_lock,
    )


def run(args: argparse.Namespace) -> Path:
    detections_path = Path(args.detections).resolve()
    config = _build_config(args)
    meta = json.loads(detections_path.read_text()).get("meta", {})
    total_frames = int(meta.get("total_frames") or 0)
    if args.relock_max <= 0:
        config.relock_max = total_frames or 999999

    profile = TargetProfile(
        name=args.name,
        number=args.number,
        primary_color=args.primary_color,
        secondary_color=args.secondary_color,
    )

    if args.photo is None:
        print("No --photo: Gatekeeper + Accumulator jersey-only mode (face_weight=0).")

    print("Running identification ...")
    result = run_identify(
        detections_path,
        config=config,
        profile=profile,
        photo=args.photo,
        max_frames=args.max_frames,
    )

    out_path = detections_path.parent / "identified.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")
    print(f"Summary: {result['summary']}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identify a single player using Gatekeeper + Accumulator pattern."
    )
    parser.add_argument("detections", type=Path, help="Path to detections.json")
    parser.add_argument(
        "--photo",
        type=Path,
        default=None,
        help="Reference face photo (optional; omit for jersey/color-only lock)",
    )
    parser.add_argument("--name", required=True, help="Player name")
    parser.add_argument("--number", type=int, required=True, help="Jersey number")
    parser.add_argument("--primary-color", required=True, help="Primary jersey hex color")
    parser.add_argument("--secondary-color", required=True, help="Secondary jersey hex color")

    parser.add_argument("--face-weight", type=float, default=2.0)
    parser.add_argument("--num-weight", type=float, default=1.5)
    parser.add_argument("--name-weight", type=float, default=0.8)
    parser.add_argument("--color-weight", type=float, default=0.2)
    parser.add_argument("--lock-threshold", type=float, default=1.8)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--initial-frames", type=int, default=30)
    parser.add_argument("--initial-face-score", type=float, default=3.0)
    parser.add_argument("--relock-window", type=int, default=5)
    parser.add_argument("--relock-cooldown", type=int, default=10)
    parser.add_argument("--relock-max", type=int, default=0, help="Max lost frames before give up (0 = whole video)")
    parser.add_argument("--ocr-every", type=int, default=5)
    parser.add_argument("--ocr-votes", type=float, default=1.5)
    parser.add_argument("--ocr-reads", type=int, default=3)
    parser.add_argument("--name-min-sim", type=float, default=0.6)
    parser.add_argument("--color-gate", type=float, default=0.1)
    parser.add_argument("--color-tolerance", type=int, default=10)
    parser.add_argument("--absence-frames", type=int, default=30)
    parser.add_argument("--number-min-conf", type=float, default=0.75)
    parser.add_argument("--zero-face-unlock", type=int, default=30)
    parser.add_argument("--relock-face-score", type=float, default=2.0)
    parser.add_argument("--handoff-face-min", type=float, default=0.45)
    parser.add_argument("--handoff-margin", type=float, default=0.25)
    parser.add_argument("--handoff-max-frames", type=int, default=60)

    parser.add_argument(
        "--kit-hist-gate",
        type=float,
        default=0.35,
        help="Stage 1: minimum kit histogram correlation before classifier runs",
    )
    parser.add_argument(
        "--min-evidence-reads",
        type=int,
        default=3,
        help="Stage 2: independent high-confidence reads required to confirm a track",
    )
    parser.add_argument(
        "--evidence-window-frames",
        type=int,
        default=90,
        help="Stage 2: temporal window for evidence accumulation (~3s at 30fps)",
    )
    parser.add_argument(
        "--min-read-spacing",
        type=int,
        default=5,
        help="Stage 2: minimum frames between independent jersey reads",
    )
    parser.add_argument(
        "--high-conf-classifier",
        type=float,
        default=0.75,
        help="Stage 2: minimum classifier confidence to count as evidence",
    )
    parser.add_argument(
        "--high-conf-ocr",
        type=float,
        default=0.55,
        help="Stage 2: minimum OCR confidence to count as evidence",
    )
    parser.add_argument(
        "--target-number-min-conf",
        type=float,
        default=0.25,
        help="Stage 2: minimum target-number classifier probability for evidence",
    )
    parser.add_argument(
        "--kit-pixel-gate",
        type=float,
        default=0.12,
        help="Stage 1: minimum primary-color torso pixel match to pass kit gate",
    )
    parser.add_argument(
        "--kit-pixel-tolerance",
        type=int,
        default=40,
        help="Stage 1: BGR tolerance for primary kit pixel match on torso crop",
    )
    parser.add_argument(
        "--allow-face-only-lock",
        action="store_true",
        help="Allow face-based lock without jersey confirmation (requires --photo)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only the first N frames from detections.json",
    )

    args = parser.parse_args()

    if not args.detections.is_file():
        raise SystemExit(f"Detections file not found: {args.detections}")
    if args.photo is not None and not args.photo.is_file():
        raise SystemExit(f"Photo not found: {args.photo}")

    for label, hex_color in (
        ("primary", args.primary_color),
        ("secondary", args.secondary_color),
    ):
        try:
            from color_score import _hex_to_rgb

            _hex_to_rgb(hex_color)
        except ValueError as exc:
            raise SystemExit(f"Invalid {label} color: {exc}") from exc

    if args.photo is None:
        if args.primary_color.lower() in {"#ffffff", "#fff"} and args.secondary_color.lower() in {
            "#000000",
            "#000",
        }:
            print(
                "Warning: white/black kit colors are generic — use real team colors for histogram gate.",
                flush=True,
            )

    run(args)


if __name__ == "__main__":
    main()
