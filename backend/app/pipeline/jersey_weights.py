"""Resolve jersey classifier checkpoint paths (shared with detetction_test/weights/)."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DET_WEIGHTS = _REPO_ROOT / "detetction_test" / "weights"
_BACKEND_WEIGHTS = Path(__file__).resolve().parents[2] / "weights"

JERSEY_NUMBER_DEFAULT = _DET_WEIGHTS / "jersey_number_b0.pt"
JERSEY_NUMBER_META_DEFAULT = _DET_WEIGHTS / "jersey_number_b0.json"


def resolve_jersey_weights(explicit: str | None = None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None
    env = os.environ.get("JERSEY_WEIGHTS", "").strip()
    if env:
        path = Path(env).expanduser()
        return path if path.is_file() else None
    for candidate in (
        JERSEY_NUMBER_DEFAULT,
        _BACKEND_WEIGHTS / "jersey_number_b0.pt",
    ):
        if candidate.is_file():
            return candidate
    return None


def resolve_jersey_meta(weights_path: Path | None) -> Path | None:
    """Sidecar JSON must sit next to the checkpoint (no cross-path fallback)."""
    if weights_path is None:
        return None
    sidecar = weights_path.with_suffix(".json")
    return sidecar if sidecar.is_file() else None
