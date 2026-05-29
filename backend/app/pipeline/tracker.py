from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .detector import BBoxScore

TrackOut = dict[str, Any]


def _detections_to_boxes(
    detections: list[BBoxScore],
    frame_shape: tuple[int, ...],
) -> Any:
    """Convert (bbox, score) pairs to ultralytics Boxes for BYTETracker."""
    from ultralytics.engine.results import Boxes

    h, w = frame_shape[:2]
    if not detections:
        data = np.zeros((0, 6), dtype=np.float32)
    else:
        rows = [
            [x1, y1, x2, y2, score, 0.0]
            for (x1, y1, x2, y2), score in detections
        ]
        data = np.asarray(rows, dtype=np.float32)
    return Boxes(data, orig_shape=(h, w)).numpy()


class PlayerTracker:
    """ByteTrack multi-object tracker for person detections."""

    def __init__(
        self,
        track_buffer: int | None = None,
        track_high_thresh: float | None = None,
        track_low_thresh: float | None = None,
        new_track_thresh: float | None = None,
        match_thresh: float | None = None,
    ) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace

        self.tracker = BYTETracker(
            args=IterableSimpleNamespace(
                tracker_type="bytetrack",
                track_buffer=int(track_buffer or os.environ.get("TRACK_BUFFER", "30")),
                track_high_thresh=float(
                    track_high_thresh or os.environ.get("TRACK_HIGH_THRESH", "0.25")
                ),
                track_low_thresh=float(
                    track_low_thresh or os.environ.get("TRACK_LOW_THRESH", "0.1")
                ),
                new_track_thresh=float(
                    new_track_thresh or os.environ.get("NEW_TRACK_THRESH", "0.15")
                ),
                match_thresh=float(match_thresh or os.environ.get("MATCH_THRESH", "0.8")),
                fuse_score=False,
            )
        )

    def update(self, detections: list[BBoxScore], frame: NDArray[np.uint8]) -> list[TrackOut]:
        boxes = _detections_to_boxes(detections, frame.shape)
        tracks = self.tracker.update(boxes, img=frame)
        out: list[TrackOut] = []
        for row in tracks:
            l, t, r, b = (float(v) for v in row[:4])
            out.append(
                {
                    "track_id": int(row[4]),
                    "bbox": [l, t, r, b],
                    "score": float(row[5]),
                }
            )
        return out
