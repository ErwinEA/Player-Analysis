"""Resolve jersey classifier checkpoint paths under backend/weights/."""

from __future__ import annotations

import os
from pathlib import Path

from backend.app.pipeline.weight_paths import BACKEND_WEIGHTS, JERSEY_CHECKPOINT

JERSEY_NUMBER_DEFAULT = BACKEND_WEIGHTS / JERSEY_CHECKPOINT
JERSEY_NUMBER_META_DEFAULT = BACKEND_WEIGHTS / "jersey_number_b0.json"


def resolve_jersey_weights(explicit: str | None = None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None
    env = os.environ.get("JERSEY_WEIGHTS", "").strip()
    if env:
        path = Path(env).expanduser()
        return path if path.is_file() else None
    if JERSEY_NUMBER_DEFAULT.is_file():
        return JERSEY_NUMBER_DEFAULT
    return None


def resolve_jersey_meta(weights_path: Path | None) -> Path | None:
    """Sidecar JSON must sit next to the checkpoint (no cross-path fallback)."""
    if weights_path is None:
        return None
    sidecar = weights_path.with_suffix(".json")
    return sidecar if sidecar.is_file() else None
