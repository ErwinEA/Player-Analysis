#!/usr/bin/env python3
"""Download and unpack SoccerNet jersey-2023 tracklet data."""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

from common import default_data_root, jersey_task_dir, load_config


def _unpack_zips(task_dir: Path, verbose: bool = True) -> list[Path]:
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
    parser = argparse.ArgumentParser(description="Download SoccerNet jersey-2023 data.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--split",
        nargs="+",
        default=["train", "test", "challenge"],
        help="Splits to download (default: train test challenge)",
    )
    parser.add_argument("--unpack", action="store_true", help="Extract zip files after download")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = args.data_root or Path(cfg["data_root"])
    data_root.mkdir(parents=True, exist_ok=True)

    password = os.environ.get("SOCCERNET_PASS") or os.environ.get("SN_PASS")
    if not password:
        print(
            "SoccerNet password required.\n"
            "Set SOCCERNET_PASS (NDA dataset password), then re-run.",
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
    print(f"Task: jersey-2023  splits: {args.split}")

    downloader.downloadDataTask(
        task="jersey-2023",
        split=args.split,
        password=password,
        verbose=args.verbose,
    )

    task_dir = jersey_task_dir(data_root)
    expected = [f"{sp}.zip" for sp in args.split if sp in {"train", "test", "challenge"}]
    missing = [
        str(task_dir / name)
        for name in expected
        if not (task_dir / name).is_file() or (task_dir / name).stat().st_size < 1024
    ]
    if missing:
        print("\nERROR: missing or empty downloads:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        sys.exit(1)

    for name in expected:
        zp = task_dir / name
        print(f"  ok: {zp} ({zp.stat().st_size / 1e6:.1f} MB)")

    if args.unpack:
        _unpack_zips(task_dir, verbose=args.verbose)

    print(f"\nDone. Jersey task dir: {task_dir}")
    print("Next: python prepare_manifest.py && python train_jersey.py")


if __name__ == "__main__":
    main()
