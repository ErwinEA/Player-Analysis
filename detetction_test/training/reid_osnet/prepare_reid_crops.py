#!/usr/bin/env python3
"""Optional: symlink SoccerNet ReID PNGs into player_crops/{person_uid}/{jersey}/.

Training reads reid/train/train/*.png directly — this step is not required.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from common import load_config
from soccernet_reid_dataset import parse_reid_filename


def _reid_train_root(data_root: Path) -> Path | None:
    for rel in ("reid/train/train", "reid/train"):
        root = data_root / rel
        if root.is_dir():
            return root
    return None


def prepare_reid_crops(
    data_root: Path,
    crops_dir: Path,
    max_images: int | None = None,
    copy: bool = False,
) -> dict:
    """Symlink/copy images to player_crops/{person_uid}/{jersey}/ using official filenames."""
    train_root = _reid_train_root(data_root)
    if train_root is None:
        print(
            f"No ReID train folder under {data_root / 'reid'}.\n"
            "Run: python download_soccernet.py --tasks reid --split train --unpack",
            file=sys.stderr,
        )
        return {"players": 0, "images": 0}

    crops_dir.mkdir(parents=True, exist_ok=True)
    player_ids: set[str] = set()
    image_count = 0

    print(f"Indexing {train_root} (optional cache; train can skip this step)...")
    for img_path in train_root.glob("*/*/*/*/*.png"):
        info = parse_reid_filename(img_path.name)
        if info is None or not info["clazz"].startswith("Player"):
            continue

        person_uid = info["person_uid"]
        jersey = info["jersey"]
        out_dir = crops_dir / person_uid / jersey
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / img_path.name
        if dest.exists():
            continue
        if copy:
            shutil.copy2(img_path, dest)
        else:
            try:
                dest.symlink_to(img_path.resolve())
            except OSError:
                shutil.copy2(img_path, dest)

        player_ids.add(person_uid)
        image_count += 1
        if max_images is not None and image_count >= max_images:
            break

    summary = {
        "players": len(player_ids),
        "images": image_count,
        "train_root": str(train_root),
        "note": "Optional; train_osnet.py reads reid/train/train directly.",
    }
    (crops_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optional cache: player_crops from SoccerNet ReID (not required for training)."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--crops-dir", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinks (safer on Windows)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = Path(args.data_root or cfg["data_root"])
    crops_dir = Path(args.crops_dir or cfg["crops_dir"])

    print(f"Preparing crops (optional): {data_root} -> {crops_dir}")
    summary = prepare_reid_crops(data_root, crops_dir, args.max_images, copy=args.copy)
    print(f"Done: {summary}")


if __name__ == "__main__":
    main()
