"""Hard scene-cut detection for broadcast footage.

Broadcast highlights interleave tactical wide shots with replays, close-ups, and
graphics. Each hard cut invalidates the pitch homography and the player tracker's
motion assumptions, so the pipeline must detect cuts to reset tracker/lock state and
suppress ball events for a short window afterwards.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class SceneCutDetector:
    """Detect hard cuts via grayscale-histogram correlation between consecutive frames.

    Cheap enough to run every frame ahead of detection/tracking. After a cut, the
    ``is_suppressed`` window stays True for ``suppress_frames`` so ball events are not
    emitted while the new shot stabilizes and a fresh lock is re-acquired.
    """

    def __init__(
        self,
        threshold: float | None = None,
        suppress_frames: int | None = None,
    ) -> None:
        self.threshold = (
            threshold if threshold is not None else _env_float("SCENE_CUT_THRESH", 0.65)
        )
        self.suppress_frames = (
            suppress_frames
            if suppress_frames is not None
            else _env_int("SCENE_CUT_SUPPRESS_FRAMES", 45)
        )
        self._prev_hist: np.ndarray | None = None
        # Start "not suppressed" so the first shot is processed normally.
        self._frames_since_cut = self.suppress_frames

    def update(self, frame: np.ndarray) -> bool:
        """Return True if this frame is a hard cut from the previous one."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)

        is_cut = False
        if self._prev_hist is not None:
            correlation = cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CORREL)
            if correlation < self.threshold:
                is_cut = True
                self._frames_since_cut = 0

        self._prev_hist = hist
        self._frames_since_cut += 1
        return is_cut

    @property
    def is_suppressed(self) -> bool:
        """True for ``suppress_frames`` frames after a detected cut."""
        return self._frames_since_cut <= self.suppress_frames
