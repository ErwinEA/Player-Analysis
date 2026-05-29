#!/usr/bin/env python3
"""Download SoccerNet data via official SoccerNet.Downloader API.

Uses downloadDataTask() for task zips (tracking, reid) — NOT downloadGames(files=[task]).
Optional: broadcast videos (720p mkv) or SoccerNet-Tracking raw video zip.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

from common import load_config

# Tasks that must use downloadDataTask (zip archives under data_root/<task>/)
ZIP_TASKS = frozenset(
    {
        "tracking",
        "reid",
        "calibration",
        "action-spotting",
        "caption",
        "dense-extraction",
        "jersey-number-recognition",
    }
)


def _expected_zips(task: str, splits: list[str]) -> list[str]:
    """Zip filenames downloadDataTask should create for a task/split combo."""
    if task == "tracking":
        allowed = {"train", "test", "challenge"}
    else:
        allowed = {"train", "valid", "test", "challenge", "test_labels", "challenge_labels"}
    return [f"{sp}.zip" for sp in splits if sp in allowed]


def _verify_task_downloads(task_dir: Path, expected: list[str]) -> list[str]:
    """Return list of missing or empty zip paths."""
    missing: list[str] = []
    for name in expected:
        path = task_dir / name
        if not path.is_file() or path.stat().st_size < 1024:
            missing.append(str(path))
    return missing


def _unpack_zips(task_dir: Path, verbose: bool = True) -> list[Path]:
    """Extract any .zip files in task_dir; skip if extract folder already exists."""
    extracted: list[Path] = []
    if not task_dir.is_dir():
        return extracted
    for zip_path in sorted(task_dir.glob("*.zip")):
        out_dir = task_dir / zip_path.stem
        if out_dir.is_dir() and any(out_dir.iterdir()):
            if verbose:
                print(f"  skip unpack (exists): {out_dir}")
            extracted.append(out_dir)
            continue
        if verbose:
            print(f"  unpacking {zip_path.name} -> {out_dir}/")
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        extracted.append(out_dir)
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download SoccerNet task data (zips) and optional videos."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["reid"],
        help="Task names for downloadDataTask (e.g. reid, tracking). Default: reid only (no videos).",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        default=["train"],
        help="Splits to download (default: train only — smaller). Use train valid test",
    )
    parser.add_argument(
        "--unpack",
        action="store_true",
        help="Extract downloaded .zip files after download",
    )
    parser.add_argument(
        "--videos",
        choices=["none", "720p", "lq", "tracking-raw"],
        default="none",
        help="Optional video download: 720p broadcast, LQ mkv, or SoccerNet-Tracking zip",
    )
    parser.add_argument(
        "--video-split",
        nargs="+",
        default=["train"],
        help="Split for --videos 720p/lq (default: train)",
    )
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = args.data_root or Path(cfg["data_root"])
    data_root.mkdir(parents=True, exist_ok=True)

    password = os.environ.get("SOCCERNET_PASS") or os.environ.get("SN_PASS")
    if not password:
        print(
            "SoccerNet password required.\n"
            "Set SOCCERNET_PASS (or SN_PASS), then re-run.\n"
            f"Data root: {data_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError as exc:
        print(f"Install SoccerNet: pip install SoccerNet\n{exc}", file=sys.stderr)
        sys.exit(1)

    downloader = SoccerNetDownloader(LocalDirectory=str(data_root))
    downloader.password = password

    print(f"Data root: {data_root}")
    print(f"Splits: {args.split}")
    print(
        "Note: SOCCERNET_PASS must be the dataset password from your SoccerNet NDA email,\n"
        "      not your website login password (401 Unauthorized usually means wrong password)."
    )

    failed = False
    for task in args.tasks:
        if task not in ZIP_TASKS:
            print(
                f"Warning: unknown zip task '{task}'. Known: {sorted(ZIP_TASKS)}",
                file=sys.stderr,
            )
        print(f"\n=== downloadDataTask: {task} ===")
        downloader.downloadDataTask(
            task=task,
            split=args.split,
            password=password,
            verbose=args.verbose,
        )
        task_dir = data_root / task
        expected = _expected_zips(task, args.split)
        missing = _verify_task_downloads(task_dir, expected)
        if missing:
            failed = True
            print("\nERROR: download did not produce expected files:", file=sys.stderr)
            for path in missing:
                print(f"  missing: {path}", file=sys.stderr)
            print(
                "\nIf you saw 'HTTP Error 401: Unauthorized' above:\n"
                "  1. Register at https://www.soccer-net.org/data and sign the NDA\n"
                "  2. Use the password from the approval / download email as SOCCERNET_PASS\n"
                "  3. Re-run: export SOCCERNET_PASS='...' && python download_soccernet.py ...\n"
                "The SDK message about 'train.zip was not uploaded' is misleading — 401 means bad password.",
                file=sys.stderr,
            )
            continue

        for name in expected:
            zp = task_dir / name
            print(f"  ok: {zp} ({zp.stat().st_size / 1e6:.1f} MB)")

        if args.unpack:
            _unpack_zips(task_dir, verbose=args.verbose)

    if args.videos == "720p":
        print("\n=== Broadcast videos (720p) — large download ===")
        downloader.downloadGames(
            files=["1_720p.mkv", "2_720p.mkv"],
            split=args.video_split,
            task="spotting",
            verbose=args.verbose,
        )
    elif args.videos == "lq":
        print("\n=== Broadcast videos (LQ mkv) — large download ===")
        downloader.downloadGames(
            files=["1.mkv", "2.mkv"],
            split=args.video_split,
            task="spotting",
            verbose=args.verbose,
        )
    elif args.videos == "tracking-raw":
        print("\n=== SoccerNet-Tracking raw video zip ===")
        downloader.downloadRAWVideo(dataset="SoccerNet-Tracking", verbose=args.verbose)

    if failed:
        sys.exit(1)

    print("\nDone. Verify with:")
    print(f'  find "{data_root}" \\( -name "*.zip" -o -name "*.mkv" -o -name "*.jpg" \\) | head -20')
    if "reid" in args.tasks:
        print(f"  python train_osnet.py")
    if "tracking" in args.tasks and args.videos != "none":
        print(f"  python extract_crops.py --data-root {data_root}")


if __name__ == "__main__":
    main()
