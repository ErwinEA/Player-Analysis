"""Resolve MobileSAM checkpoint paths."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BACKEND_WEIGHTS = Path(__file__).resolve().parents[2] / "weights"

DEFAULT_SAM_CHECKPOINT = "mobile_sam.pt"


def resolve_sam_weights(explicit: str | None = None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None
    env = os.environ.get("SAM_WEIGHTS", "").strip()
    if env:
        path = Path(env).expanduser()
        return path if path.is_file() else None
    for candidate in (
        _BACKEND_WEIGHTS / DEFAULT_SAM_CHECKPOINT,
        _REPO_ROOT / "detetction_test" / "weights" / DEFAULT_SAM_CHECKPOINT,
    ):
        if candidate.is_file():
            return candidate
    return None
