"""Shared paths and config loading for OSNet training scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

REID_ROOT = Path(__file__).resolve().parent
DETECTION_TEST_ROOT = REID_ROOT.parent.parent


def default_data_root() -> Path:
    env = os.environ.get("SOCCERNET_DATA")
    if env:
        return Path(env)
    return DETECTION_TEST_ROOT / "data" / "soccernet"


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or REID_ROOT / "config.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    if not cfg.get("data_root"):
        cfg["data_root"] = str(default_data_root())
    if not cfg.get("crops_dir"):
        cfg["crops_dir"] = str(Path(cfg["data_root"]) / "player_crops")
    cfg["output_dir"] = str(REID_ROOT / cfg.get("output_dir", "outputs"))
    weights_out = cfg.get("weights_out", "../../weights/osnet_x1_0_soccernet.pth")
    cfg["weights_out"] = str((REID_ROOT / weights_out).resolve())
    return cfg


def train_manifest_path(output_dir: Path) -> Path:
    return output_dir / "train_manifest.json"


def save_train_manifest(output_dir: Path, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = train_manifest_path(output_dir)
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_train_manifest(output_dir: Path) -> dict | None:
    path = train_manifest_path(output_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text())
