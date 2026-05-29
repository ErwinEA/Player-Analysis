from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.app.pipeline.run import run_pipeline
from backend.app.schemas import PlayerDetails


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run player tracking pipeline on a local video (no HTTP server).",
    )
    parser.add_argument("video", type=Path, help="Path to MP4/MOV video file")
    parser.add_argument("details", type=Path, help="Path to PlayerDetails JSON file")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    args = parser.parse_args()

    if not args.video.is_file():
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 1
    if not args.details.is_file():
        print(f"Details JSON not found: {args.details}", file=sys.stderr)
        return 1

    details_data = json.loads(args.details.read_text(encoding="utf-8"))
    details = PlayerDetails.model_validate(details_data)

    result = run_pipeline(str(args.video), details)
    indent = 2 if args.pretty else None
    print(result.model_dump_json(indent=indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
