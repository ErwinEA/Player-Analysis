from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    sim = float(np.dot(a, b) / (norm_a * norm_b))
    return max(0.0, min(1.0, sim))


class FaceMatcher:
    """Reference face embedding + per-bbox cosine similarity."""

    def __init__(self, reference_photo_path: str | Path) -> None:
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

        photo_path = Path(reference_photo_path)
        img = cv2.imread(str(photo_path))
        if img is None:
            raise ValueError(f"Could not read reference photo: {photo_path}")

        faces = self.app.get(img)
        if not faces:
            raise ValueError(f"No face found in reference photo: {photo_path}")

        self.ref_embedding = np.asarray(faces[0].embedding, dtype=np.float32)

    def score_bbox(self, frame: NDArray, bbox: list[float]) -> float:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0

        faces = self.app.get(crop)
        if not faces:
            return 0.0

        best = 0.0
        for face in faces:
            emb = np.asarray(face.embedding, dtype=np.float32)
            sim = _cosine_similarity(self.ref_embedding, emb)
            best = max(best, sim)
        return best

    def score_frame(
        self, frame: NDArray, people: list[dict]
    ) -> dict[int, float]:
        return {int(p["id"]): self.score_bbox(frame, p["bbox"]) for p in people}


class NullFaceMatcher:
    """No-op face matcher when --photo is omitted."""

    def score_bbox(self, frame: NDArray, bbox: list[float]) -> float:
        return 0.0

    def score_frame(self, frame: NDArray, people: list[dict]) -> dict[int, float]:
        return {}
