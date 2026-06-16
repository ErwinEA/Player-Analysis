"""Root-cause instrumentation for target lock acquisition failures."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def lock_audit_enabled() -> bool:
    return os.environ.get("LOCK_AUDIT", "0").strip().lower() in ("1", "true", "yes")


def lock_audit_dir() -> Path:
    raw = os.environ.get("LOCK_AUDIT_DIR", "debug_ocr")
    return Path(raw).resolve()


@dataclass
class TrackAuditStats:
    track_id: int
    frames_seen: int = 0
    ocr_attempts: int = 0
    ocr_reads: int = 0
    ocr_matches_target: int = 0
    match_successes: int = 0
    match_rejections: int = 0
    lock_candidates: int = 0
    best_conf_target: float = 0.0
    best_read: int | None = None


@dataclass
class LockAudit:
    """Structured OCR + lock diagnostics (enable with LOCK_AUDIT=1)."""

    target_jersey: int
    enabled: bool = field(init=False)
    root: Path = field(init=False)
    tracks: dict[int, TrackAuditStats] = field(default_factory=dict)
    ocr_attempts: int = 0
    ocr_reads: int = 0
    ocr_reads_target: int = 0
    successful_matches: int = 0
    lock_candidates: int = 0
    locks_created: int = 0
    lock_resets: int = 0
    lock_failures: int = 0
    _image_count: int = 0

    def __post_init__(self) -> None:
        self.enabled = lock_audit_enabled()
        self.root = lock_audit_dir()
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            started = datetime.now(timezone.utc).isoformat()
            with (self.root / "lock_audit.log").open("a", encoding="utf-8") as f:
                f.write(f"# Lock audit session started {started}\n")

    def _track(self, track_id: int) -> TrackAuditStats:
        tid = int(track_id)
        if tid not in self.tracks:
            self.tracks[tid] = TrackAuditStats(track_id=tid)
        return self.tracks[tid]

    def record_track_seen(self, track_id: int) -> None:
        if not self.enabled:
            return
        self._track(track_id).frames_seen += 1

    def log_ocr_attempt(
        self,
        *,
        frame_idx: int,
        track_id: int,
        ocr_text: str,
        ocr_conf: float,
        parsed_number: int | None,
        frame: NDArray | None = None,
        bbox: list[float] | None = None,
    ) -> None:
        if not self.enabled:
            return
        self.ocr_attempts += 1
        ts = self._track(track_id)
        ts.ocr_attempts += 1
        if parsed_number is not None and ocr_conf > 0:
            self.ocr_reads += 1
            ts.ocr_reads += 1
            ts.best_read = parsed_number
            ts.best_conf_target = max(ts.best_conf_target, ocr_conf)
        if parsed_number == self.target_jersey:
            self.ocr_reads_target += 1
            ts.ocr_matches_target += 1

        logger.info(
            "[OCR] frame=%s track=%s text='%s' conf=%.2f",
            frame_idx,
            track_id,
            ocr_text,
            ocr_conf,
        )
        self._save_debug_image(
            frame_idx=frame_idx,
            track_id=track_id,
            lines=[
                f"Frame: {frame_idx}",
                f"Track: {track_id}",
                f'OCR: "{ocr_text}"',
                f"Conf: {ocr_conf:.2f}",
                f"Parsed: {parsed_number}",
            ],
            frame=frame,
            bbox=bbox,
        )

    def log_match_check(
        self,
        *,
        frame_idx: int,
        track_id: int,
        ocr_text: str,
    ) -> None:
        if not self.enabled:
            return
        logger.info(
            "[MATCH_CHECK] frame=%s track=%s text='%s' target='%s'",
            frame_idx,
            track_id,
            ocr_text,
            self.target_jersey,
        )

    def log_match_success(
        self,
        *,
        frame_idx: int,
        track_id: int,
        ocr_text: str,
        ocr_conf: float,
    ) -> None:
        if not self.enabled:
            return
        self.successful_matches += 1
        self._track(track_id).match_successes += 1
        logger.info(
            "[MATCH_SUCCESS] frame=%s track=%s text='%s' conf=%.2f",
            frame_idx,
            track_id,
            ocr_text,
            ocr_conf,
        )

    def log_match_rejected(
        self,
        *,
        frame_idx: int,
        track_id: int,
        reason: str,
        ocr_text: str = "",
        ocr_conf: float = 0.0,
    ) -> None:
        if not self.enabled:
            return
        self._track(track_id).match_rejections += 1
        logger.info(
            "[MATCH_REJECTED] frame=%s track=%s reason=%s text='%s' conf=%.2f",
            frame_idx,
            track_id,
            reason,
            ocr_text,
            ocr_conf,
        )

    def log_lock_candidate(
        self,
        *,
        frame_idx: int,
        track_id: int,
        ocr_text: str,
        ocr_conf: float,
        method: str = "",
    ) -> None:
        if not self.enabled:
            return
        self.lock_candidates += 1
        self._track(track_id).lock_candidates += 1
        extra = f" method={method}" if method else ""
        logger.info(
            "[LOCK_CANDIDATE] frame=%s track=%s text='%s' conf=%.2f%s",
            frame_idx,
            track_id,
            ocr_text,
            ocr_conf,
            extra,
        )

    def log_lock_created(
        self,
        *,
        frame_idx: int,
        track_id: int,
        method: str,
        confidence: float,
    ) -> None:
        if not self.enabled:
            return
        self.locks_created += 1
        logger.info(
            "[LOCK_CREATED] frame=%s track=%s method=%s conf=%.2f",
            frame_idx,
            track_id,
            method,
            confidence,
        )

    def log_lock_failed(
        self,
        *,
        frame_idx: int,
        track_id: int | None,
        reason: str,
    ) -> None:
        if not self.enabled:
            return
        self.lock_failures += 1
        tid = track_id if track_id is not None else "-"
        logger.info(
            "[LOCK_FAILED] frame=%s track=%s reason=%s",
            frame_idx,
            tid,
            reason,
        )

    def log_lock_reset(self, *, frame_idx: int, reason: str) -> None:
        if not self.enabled:
            return
        self.lock_resets += 1
        logger.info("[LOCK_RESET] frame=%s reason=%s", frame_idx, reason)

    def evaluate_ocr_match(
        self,
        *,
        frame_idx: int,
        track_id: int,
        parsed_number: int | None,
        ocr_conf: float,
        ocr_text: str,
        number_min_conf: float,
    ) -> None:
        """Classify an OCR read against the target jersey."""
        if not self.enabled:
            return
        self.log_match_check(
            frame_idx=frame_idx,
            track_id=track_id,
            ocr_text=ocr_text,
        )
        if parsed_number is None or ocr_conf <= 0:
            self.log_match_rejected(
                frame_idx=frame_idx,
                track_id=track_id,
                reason="no_read",
                ocr_text=ocr_text,
                ocr_conf=ocr_conf,
            )
            return
        if parsed_number != self.target_jersey:
            self.log_match_rejected(
                frame_idx=frame_idx,
                track_id=track_id,
                reason="wrong_number",
                ocr_text=ocr_text,
                ocr_conf=ocr_conf,
            )
            return
        if ocr_conf < number_min_conf:
            self.log_match_rejected(
                frame_idx=frame_idx,
                track_id=track_id,
                reason="low_confidence",
                ocr_text=ocr_text,
                ocr_conf=ocr_conf,
            )
            return
        self.log_match_success(
            frame_idx=frame_idx,
            track_id=track_id,
            ocr_text=ocr_text,
            ocr_conf=ocr_conf,
        )

    def _save_debug_image(
        self,
        *,
        frame_idx: int,
        track_id: int,
        lines: list[str],
        frame: NDArray | None,
        bbox: list[float] | None,
    ) -> None:
        if frame is None or bbox is None:
            return
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return
        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return
        pad = 70
        canvas = np.zeros((crop.shape[0] + pad, crop.shape[1], 3), dtype=np.uint8)
        canvas[pad:, :] = crop
        y = 18
        for line in lines:
            cv2.putText(
                canvas,
                line,
                (4, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            y += 16
        path = self.root / f"frame_{frame_idx:06d}_track_{track_id}.jpg"
        cv2.imwrite(str(path), canvas)
        self._image_count += 1

    def track_summary_lines(self) -> list[str]:
        lines: list[str] = []
        for tid in sorted(self.tracks):
            ts = self.tracks[tid]
            lines.append(
                f"Track {tid}: frames_seen={ts.frames_seen} "
                f"ocr_attempts={ts.ocr_attempts} ocr_reads={ts.ocr_reads} "
                f'reads_of_"{self.target_jersey}"={ts.ocr_matches_target} '
                f"match_successes={ts.match_successes} "
                f"best_read={ts.best_read} best_conf={ts.best_conf_target:.2f}"
            )
        return lines

    def report(self) -> dict[str, Any]:
        return {
            "target_jersey": self.target_jersey,
            "ocr_attempts": self.ocr_attempts,
            "ocr_reads": self.ocr_reads,
            f"ocr_reads_{self.target_jersey}": self.ocr_reads_target,
            "successful_matches": self.successful_matches,
            "lock_candidates": self.lock_candidates,
            "locks_created": self.locks_created,
            "lock_resets": self.lock_resets,
            "lock_failures": self.lock_failures,
            "debug_images_saved": self._image_count,
            "tracks": {
                str(tid): {
                    "frames_seen": ts.frames_seen,
                    "ocr_attempts": ts.ocr_attempts,
                    "ocr_reads": ts.ocr_reads,
                    "ocr_matches_target": ts.ocr_matches_target,
                    "match_successes": ts.match_successes,
                    "match_rejections": ts.match_rejections,
                    "lock_candidates": ts.lock_candidates,
                    "best_read": ts.best_read,
                    "best_conf_target": round(ts.best_conf_target, 4),
                }
                for tid, ts in sorted(self.tracks.items())
            },
        }

    def write_report(self, path: Path | None = None) -> Path:
        out = path or (self.root / "lock_audit_report.json")
        payload = self.report()
        out.write_text(json.dumps(payload, indent=2))
        if self.enabled:
            for line in self.track_summary_lines():
                logger.info("[TRACK_SUMMARY] %s", line)
            logger.info("[LOCK_AUDIT_REPORT] %s", json.dumps(payload))
        return out
