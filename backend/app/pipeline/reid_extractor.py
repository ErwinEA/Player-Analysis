"""OSNet feature extractor: BGR crop -> 512-d L2-normalized embedding."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from backend.app.pipeline.weight_paths import (
    BACKEND_WEIGHTS,
    OSNET_ALIAS,
    OSNET_CHECKPOINT,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

INPUT_H = 256
INPUT_W = 128


class ReIDFeatureExtractor:
    """Load OSNet-x1_0 and extract appearance embeddings from person crops."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        model_name: str = "osnet_x1_0",
        device: str | None = None,
    ) -> None:
        import torch
        from torchreid.utils import FeatureExtractor

        wts = weights_path or _default_reid_weights()
        if wts is None:
            primary = BACKEND_WEIGHTS / OSNET_CHECKPOINT
            alias = BACKEND_WEIGHTS / OSNET_ALIAS
            wts = str(primary if primary.is_file() else alias)

        use_gpu = os.environ.get("REID_GPU", "0") == "1" and torch.cuda.is_available()
        dev = device or ("cuda" if use_gpu else "cpu")

        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=str(wts) if Path(wts).is_file() else None,
            device=dev,
            verbose=False,
        )
        self.device = dev

    def preprocess(self, crop_bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if crop_bgr.size == 0:
            return crop_bgr
        return cv2.resize(crop_bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)

    def embed(self, crop_bgr: NDArray[np.uint8]) -> np.ndarray:
        """Return 512-d float32 L2-normalized vector."""
        if crop_bgr.size == 0:
            return np.zeros(512, dtype=np.float32)
        crop = self.preprocess(crop_bgr)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        feat = self.extractor(rgb)
        if hasattr(feat, "cpu"):
            feat = feat.cpu().numpy()
        vec = np.asarray(feat, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_batch(self, crops_bgr: list[NDArray[np.uint8]]) -> np.ndarray:
        if not crops_bgr:
            return np.zeros((0, 512), dtype=np.float32)
        return np.stack([self.embed(c) for c in crops_bgr], axis=0)


def _default_reid_weights() -> str | None:
    for candidate in (
        BACKEND_WEIGHTS / OSNET_CHECKPOINT,
        BACKEND_WEIGHTS / OSNET_ALIAS,
    ):
        if candidate.is_file():
            return str(candidate)
    return None
