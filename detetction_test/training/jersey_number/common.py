"""Shared paths and config for SoccerNet jersey-number training."""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

JERSEY_ROOT = Path(__file__).resolve().parent
DETECTION_TEST_ROOT = JERSEY_ROOT.parent.parent


def default_data_root() -> Path:
    env = os.environ.get("SOCCERNET_DATA")
    if env:
        return Path(env)
    return DETECTION_TEST_ROOT / "data" / "soccernet"


def jersey_task_dir(data_root: Path) -> Path:
    return data_root / "jersey-2023"


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or JERSEY_ROOT / "config.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    if not cfg.get("data_root"):
        cfg["data_root"] = str(default_data_root())
    cfg["output_dir"] = str(JERSEY_ROOT / cfg.get("output_dir", "outputs"))
    weights_out = cfg.get("weights_out", "../../weights/jersey_number_b0.pt")
    cfg["weights_out"] = str((JERSEY_ROOT / weights_out).resolve())
    meta_out = cfg.get("meta_out", "../../weights/jersey_number_b0.json")
    cfg["meta_out"] = str((JERSEY_ROOT / meta_out).resolve())
    return cfg


def manifest_path(output_dir: Path) -> Path:
    return output_dir / "jersey_manifest.json"


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
