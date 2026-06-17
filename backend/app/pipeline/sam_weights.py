"""Resolve MobileSAM checkpoint paths."""

from __future__ import annotations

import os
from pathlib import Path

from backend.app.pipeline.weight_paths import BACKEND_WEIGHTS, SAM_CHECKPOINT


def resolve_sam_weights(explicit: str | None = None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None
    env = os.environ.get("SAM_WEIGHTS", "").strip()
    if env:
        path = Path(env).expanduser()
        return path if path.is_file() else None
    candidate = BACKEND_WEIGHTS / SAM_CHECKPOINT
    return candidate if candidate.is_file() else None
