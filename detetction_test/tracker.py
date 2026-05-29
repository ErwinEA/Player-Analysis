from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from bbox_utils import normalize_bbox
from detect import Detection

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class TrackedPerson:
    id: int
    bbox: list[float]
    conf: float


@dataclass
class TrackedFrame:
    frame: int
    timestamp: float
    people: list[TrackedPerson]


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _best_detection_conf(track_bbox: list[float], detections: list[Detection]) -> float:
    best_conf = 0.0
    best_iou = 0.0
    for det in detections:
        iou = _iou(track_bbox, det.bbox)
        if iou > best_iou:
            best_iou = iou
            best_conf = det.conf
    return best_conf if best_iou > 0.0 else 0.0


def _detections_to_boxes(
    detections: list[Detection],
    frame_shape: tuple[int, ...],
) -> Any:
    """Convert Detection list to ultralytics Boxes for BYTETracker."""
    from ultralytics.engine.results import Boxes

    h, w = frame_shape[:2]
    if not detections:
        data = np.zeros((0, 6), dtype=np.float32)
    else:
        rows = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            rows.append([x1, y1, x2, y2, det.conf, 0.0])
        data = np.asarray(rows, dtype=np.float32)
    return Boxes(data, orig_shape=(h, w)).numpy()


class PersonTracker:
    """ByteTrack multi-object tracker for person detections."""

    def __init__(
        self,
        track_buffer: int = 30,
        track_high_thresh: float = 0.25,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.15,
        match_thresh: float = 0.8,
    ) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace

        self.track_buffer = track_buffer
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh

        self.tracker = BYTETracker(
            args=IterableSimpleNamespace(
                tracker_type="bytetrack",
                track_buffer=track_buffer,
                track_high_thresh=track_high_thresh,
                track_low_thresh=track_low_thresh,
                new_track_thresh=new_track_thresh,
                match_thresh=match_thresh,
                fuse_score=False,
            )
        )

    def update(
        self,
        detections: list[Detection],
        frame: NDArray[np.uint8],
    ) -> list[TrackedPerson]:
        boxes = _detections_to_boxes(detections, frame.shape)
        tracks = self.tracker.update(boxes, img=frame)
        people: list[TrackedPerson] = []
        for row in tracks:
            l, t, r, b = (float(v) for v in row[:4])
            bbox = normalize_bbox([l, t, r, b], frame.shape)
            people.append(
                TrackedPerson(
                    id=int(row[4]),
                    bbox=bbox,
                    conf=_best_detection_conf(bbox, detections) or float(row[5]),
                )
            )
        return people
