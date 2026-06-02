"""Resolve PyTorch / Ultralytics inference devices from environment."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def mps_available() -> bool:
    try:
        import torch

        return bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        return False


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def resolve_torch_device_str(
    pref: str | None = None,
    *,
    env_key: str = "INFERENCE_DEVICE",
    default: str = "auto",
) -> str:
    """Return ``mps``, ``cuda``, or ``cpu`` for torch modules."""
    raw = (pref or os.environ.get(env_key, default)).strip().lower()
    if raw in ("cuda", "mps", "cpu"):
        return raw
    if raw == "auto":
        if cuda_available():
            return "cuda"
        if mps_available():
            return "mps"
        return "cpu"
    return "cpu"


def resolve_ultralytics_device() -> str | int:
    """Device for Ultralytics ``predict()`` (``YOLO_DEVICE`` overrides ``INFERENCE_DEVICE``)."""
    yolo = os.environ.get("YOLO_DEVICE", "").strip().lower()
    if yolo:
        return yolo
    return resolve_torch_device_str(env_key="INFERENCE_DEVICE")


def torch_device_label(env_value: str | None, resolved: str) -> str:
    """Human-readable label for startup logs."""
    raw = (env_value or "").strip() or "(unset)"
    return f"{raw}→{resolved}"
