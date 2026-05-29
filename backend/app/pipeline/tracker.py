"""Multi-object tracker wrapper (Ultralytics ByteTrack / BoT-SORT).

Spike (Ultralytics 8.4.x): BOTSORT shares BYTETracker's ``update(boxes, img=frame)``
interface and exposes ``reset()``. GMC param is ``gmc_method`` (default sparseOptFlow,
safe on MPS/CPU; ecc needs opencv-contrib). Native ReID needs detector features and a
custom OSNet is not drop-in, so ``with_reid=False`` here and the post-hoc OSNet lock in
run.py is unchanged. Select the backend with ``TRACKER_TYPE=botsort|bytetrack``.
"""

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
        tracker_type: str | None = None,
    ) -> None:
        from ultralytics.utils import IterableSimpleNamespace

        ttype = (tracker_type or os.environ.get("TRACKER_TYPE", "bytetrack")).strip().lower()
        if ttype not in ("bytetrack", "botsort"):
            ttype = "bytetrack"
        # BoT-SORT keeps lost tracks longer (camera pans/occlusion) by default.
        default_buffer = "120" if ttype == "botsort" else "30"

        args = dict(
            tracker_type=ttype,
            track_buffer=int(track_buffer or os.environ.get("TRACK_BUFFER", default_buffer)),
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

        if ttype == "botsort":
            from ultralytics.trackers.bot_sort import BOTSORT

            args.update(
                gmc_method=os.environ.get("GMC_METHOD", "sparseOptFlow"),
                proximity_thresh=float(os.environ.get("PROXIMITY_THRESH", "0.5")),
                appearance_thresh=float(os.environ.get("APPEARANCE_THRESH", "0.25")),
                with_reid=False,
                model="auto",
            )
            self.tracker = BOTSORT(args=IterableSimpleNamespace(**args))
        else:
            from ultralytics.trackers.byte_tracker import BYTETracker

            self.tracker = BYTETracker(args=IterableSimpleNamespace(**args))

    def reset(self) -> None:
        """Clear all active/lost tracks and reinitialize. Call on scene cut so the
        new shot starts with fresh track IDs instead of associating across the cut."""
        self.tracker.reset()

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
