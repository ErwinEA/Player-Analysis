"""OSNet ReID for player lock (SoccerNet-trained weights)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from backend.app.pipeline.inference_device import resolve_torch_device_str
from backend.app.pipeline.reid_extractor import ReIDFeatureExtractor
from backend.app.pipeline.weight_paths import (
    BACKEND_WEIGHTS,
    OSNET_ALIAS,
    OSNET_CHECKPOINT,
)

logger = logging.getLogger(__name__)

_extractor: object | None = None
_extractor_unavailable = False


def resolve_reid_weights_path() -> Path | None:
    """Resolve OSNet checkpoint path (user file: osnet_x1_0_soccernet.pth)."""
    explicit = os.environ.get("REID_WEIGHTS", "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path

    for candidate in (
        BACKEND_WEIGHTS / OSNET_CHECKPOINT,
        BACKEND_WEIGHTS / OSNET_ALIAS,
    ):
        if candidate.is_file():
            return candidate
    return None


def reid_enabled() -> bool:
    return os.environ.get("REID_ENABLED", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def get_reid_extractor():
    """Lazy singleton ReIDFeatureExtractor; None if weights or torchreid missing."""
    global _extractor, _extractor_unavailable
    if _extractor_unavailable or not reid_enabled():
        return None
    if _extractor is not None:
        return _extractor

    weights = resolve_reid_weights_path()

    try:
        if weights is None:
            logger.info(
                "%s not found under backend/weights/; "
                "using ImageNet-pretrained osnet_x1_0 (set REID_WEIGHTS for SoccerNet weights)",
                OSNET_CHECKPOINT,
            )
        reid_device = resolve_torch_device_str(env_key="INFERENCE_DEVICE")
        _extractor = ReIDFeatureExtractor(
            weights_path=str(weights) if weights is not None else None,
            device=reid_device,
        )
        if weights is not None:
            logger.info("ReID loaded from %s on %s", weights, reid_device)
        else:
            logger.info("ReID loaded (pretrained osnet_x1_0) on %s", reid_device)
        return _extractor
    except Exception as exc:
        logger.warning("ReID unavailable (%s). Install torchreid + torch.", exc)
        _extractor_unavailable = True
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def mean_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(embeddings, axis=0)
    mean = stack.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean = mean / norm
    return mean.astype(np.float32)
