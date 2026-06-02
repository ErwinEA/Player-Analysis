"""Default paths for model weights under detection_test/weights/."""

from __future__ import annotations

import os
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

YOLO_SOCCER = WEIGHTS_DIR / "yolo_soccer.pt"
OSNET_SOCCER = WEIGHTS_DIR / "osnet_x1_0_soccernet.pth"
OSNET_SOCCER_ALIAS = WEIGHTS_DIR / "osnet_soccer_reid.pth"
JERSEY_NUMBER = WEIGHTS_DIR / "jersey_number_b0.pt"
JERSEY_NUMBER_META = WEIGHTS_DIR / "jersey_number_b0.json"
BALL_YOLO = WEIGHTS_DIR / "yolov8n_ball.pt"

# Fallbacks when user-supplied weights are missing
YOLO_FALLBACK = Path(__file__).resolve().parent.parent / "yolov8m.pt"
BALL_FALLBACK = Path(__file__).resolve().parent.parent / "yolov8n.pt"


def resolve_yolo_weights(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("YOLO_WEIGHTS")
    if env:
        return env
    if YOLO_SOCCER.is_file():
        return str(YOLO_SOCCER)
    return str(YOLO_FALLBACK)


def resolve_reid_weights(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit if Path(explicit).is_file() else None
    env = os.environ.get("REID_WEIGHTS")
    if env and Path(env).is_file():
        return env
    for candidate in (OSNET_SOCCER, OSNET_SOCCER_ALIAS):
        if candidate.is_file():
            return str(candidate)
    return None


def resolve_jersey_weights(explicit: str | None = None) -> Path | None:
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    env = os.environ.get("JERSEY_WEIGHTS")
    if env and Path(env).is_file():
        return Path(env)
    return JERSEY_NUMBER if JERSEY_NUMBER.is_file() else None


def resolve_ball_weights(explicit: str | None = None) -> str | None:
    if explicit and Path(explicit).is_file():
        return explicit
    env = os.environ.get("BALL_WEIGHTS")
    if env and Path(env).is_file():
        return env
    if BALL_YOLO.is_file():
        return str(BALL_YOLO)
    if BALL_FALLBACK.is_file():
        return str(BALL_FALLBACK)
    return None


def yolo_class_filter() -> list[int] | None:
    """Return COCO person class filter, or None for single-class soccer weights."""
    mode = os.environ.get("YOLO_CLASS_FILTER", "auto").lower()
    if mode == "none" or mode == "0":
        return None
    if mode == "coco":
        return [0]
    # auto: skip class filter when soccer weights file is in use
    weights = os.environ.get("YOLO_WEIGHTS", "")
    if "yolo_soccer" in weights or YOLO_SOCCER.is_file():
        return None
    return [0]
