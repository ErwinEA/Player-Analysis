#!/usr/bin/env python3
"""Build labeled jersey crop manifest from SoccerNet jersey-2023."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import jersey_task_dir, load_config, manifest_path, save_json
from soccernet_jersey_dataset import build_manifest, config_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label tracklet thumbnails: {image_path -> jersey_number} manifest."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--unpack", action="store_true", help="Unpack zips if splits missing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = args.data_root or Path(cfg["data_root"])
    task_dir = jersey_task_dir(data_root)
    output_dir = Path(cfg["output_dir"])

    if args.unpack:
        import zipfile

        for zip_path in sorted(task_dir.glob("*.zip")):
            out_dir = task_dir / zip_path.stem
            if out_dir.is_dir() and any(out_dir.iterdir()):
                continue
            print(f"Unpacking {zip_path.name} ...")
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(out_dir)

    manifest = build_manifest(task_dir, cfg)
    manifest["config_fingerprint"] = config_fingerprint(cfg)
    out = manifest_path(output_dir)
    save_json(out, manifest)

    print(f"Wrote {out}")
    print(f"Counts: {manifest['counts']}")
    print(f"Classes: {len(manifest['class_names'])} ({manifest['class_names'][:3]} ...)")


if __name__ == "__main__":
    main()
