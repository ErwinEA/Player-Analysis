"""Simple LRU-ish frame cache to avoid repeated cv2.imread across pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from numpy.typing import NDArray


class FrameCache:
    def __init__(self, frames_dir: Path, *, max_entries: int = 128) -> None:
        self.frames_dir = frames_dir
        self.max_entries = max_entries
        self._cache: dict[int, NDArray] = {}
        self._order: list[int] = []

    def get(self, frame_idx: int) -> NDArray:
        if frame_idx in self._cache:
            return self._cache[frame_idx]
        path = self.frames_dir / f"frame_{frame_idx:06d}.jpg"
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Missing frame image: {path}")
        self._cache[frame_idx] = img
        self._order.append(frame_idx)
        if len(self._order) > self.max_entries:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        return img
