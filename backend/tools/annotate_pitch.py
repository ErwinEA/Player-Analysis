#!/usr/bin/env python3
"""Interactively annotate pitch corners on a video frame and save homography calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

# Allow running as script from repo root or backend/
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.app.pipeline.pitch_homography import (  # noqa: E402
    CORNER_LABELS,
    build_calibration,
    default_calibration_dir,
    draw_corners_overlay,
    render_pitch_template,
    save_calibration,
    warp_frame_to_topview,
)


def _load_frame(video_path: Path, frame_index: int) -> tuple[cv2.typing.MatLike, tuple[int, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"Could not read frame {frame_index} from {video_path}")
        h, w = frame.shape[:2]
        return frame, (w, h)
    finally:
        cap.release()


def _interactive_corners(frame) -> list[list[float]]:
    points: list[tuple[int, int]] = []
    window = "Annotate pitch corners (TL, TR, BR, BL) — click 4 points"

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    print("Click 4 pitch landmarks in order: top_left, top_right, bottom_right, bottom_left")
    print("Use visible line intersections; extrapolate off-screen corners if needed.")
    print("Keys: r=reset, s=save, q=quit without saving")

    while True:
        display = draw_corners_overlay(frame, [[p[0], p[1]] for p in points] if points else [])
        if len(points) < 4:
            cv2.putText(
                display,
                f"Click corner {len(points) + 1}/4: {CORNER_LABELS[len(points)]}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow(window, display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("r"):
            points.clear()
        elif key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("Annotation cancelled.")
        elif key == ord("s") and len(points) == 4:
            break

    cv2.destroyAllWindows()
    return [[float(x), float(y)] for x, y in points]


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate pitch corners and compute homography.")
    parser.add_argument("--video", type=Path, required=True, help="Input video path")
    parser.add_argument("--frame", type=int, default=100, help="Frame index to annotate")
    parser.add_argument("--name", type=str, default=None, help="Calibration name (default: video stem)")
    parser.add_argument(
        "--corners",
        type=str,
        default=None,
        help='JSON file or inline JSON list of 4 [x,y] points (skips interactive UI)',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: backend/data/pitch_calibration)",
    )
    args = parser.parse_args()

    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        print(f"Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    frame, image_size = _load_frame(video_path, args.frame)
    name = args.name or video_path.stem
    out_dir = args.output_dir or default_calibration_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.corners:
        raw = args.corners
        corners_path = Path(raw)
        if corners_path.is_file():
            corners = json.loads(corners_path.read_text(encoding="utf-8"))
        else:
            corners = json.loads(raw)
    else:
        corners = _interactive_corners(frame)

    try:
        cal = build_calibration(
            name,
            corners,
            video_path=str(video_path),
            frame_index=args.frame,
            image_size=image_size,
        )
    except ValueError as exc:
        print(f"Homography failed: {exc}", file=sys.stderr)
        print(
            "Tip: click visible pitch boundary points in TL→TR→BR→BL order; "
            "extrapolated off-frame corners are OK.",
            file=sys.stderr,
        )
        sys.exit(1)

    cal_path = out_dir / f"{name}.json"
    save_calibration(cal, cal_path)

    overlay = draw_corners_overlay(frame, cal.image_corners)
    overlay_path = out_dir / f"{name}_corners.jpg"
    cv2.imwrite(str(overlay_path), overlay)

    template = render_pitch_template()
    template_path = out_dir / f"{name}_topview_template.png"
    cv2.imwrite(str(template_path), template)

    warped = warp_frame_to_topview(frame, cal)
    warped_path = out_dir / f"{name}_topview_warp.jpg"
    cv2.imwrite(str(warped_path), warped)

    print(f"Saved calibration -> {cal_path}")
    print(f"Saved corner overlay -> {overlay_path}")
    print(f"Saved UI pitch template -> {template_path}")
    print(f"Saved warped frame check -> {warped_path}")
    print(f"Homography (image->meters):\n{cal.H}")


if __name__ == "__main__":
    main()
