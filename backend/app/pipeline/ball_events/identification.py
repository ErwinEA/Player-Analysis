"""Target identification tiers for ball-event gating (lock-centric → identification-centric)."""

from __future__ import annotations

import logging
import os
from typing import Literal

from backend.app.pipeline.ball_events.timeline import TargetFrameSample

logger = logging.getLogger(__name__)

IdentificationType = Literal["locked", "ocr", "reid", "track"]
LockConfidence = Literal["strong", "weak"]

_IDENTIFIED_TYPES: frozenset[str] = frozenset({"locked", "ocr", "reid", "track"})


def _frame_log_enabled() -> bool:
    return os.environ.get("BALL_EVENTS_FRAME_LOG", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def lock_confidence_for_id_type(id_type: str | None) -> LockConfidence | None:
    if id_type == "locked":
        return "strong"
    if id_type in ("ocr", "reid", "track"):
        return "weak"
    return None


def has_target_identification(
    sample: TargetFrameSample,
    locked_at_frame: int | None,
) -> tuple[bool, LockConfidence | None, IdentificationType | None]:
    """Whether this frame may contribute to possession/events.

    Requires a pitch/localized target position (target_m) and a confirmed id tier.
    Voting-only frames (no identification_type) are rejected unless they are
    post-lock with position (backward compatible with older samples).
    """
    id_type = sample.identification_type
    if sample.target_m is None:
        return False, None, None

    if id_type in _IDENTIFIED_TYPES:
        conf = lock_confidence_for_id_type(id_type)
        return True, conf, id_type  # type: ignore[return-value]

    # Legacy rows / tests without identification_type: treat post-lock as strong.
    if locked_at_frame is not None and sample.frame >= locked_at_frame:
        return True, "strong", "locked"

    return False, None, None


def log_frame_evaluation(
    *,
    frame: int,
    sample: TargetFrameSample,
    locked_at_frame: int | None,
    possessed: bool,
    ball_speed_m_s: float | None,
    dist_m: float | None,
    event_kind: str | None,
) -> None:
    if not _frame_log_enabled():
        return
    identified, lock_conf, id_type = has_target_identification(sample, locked_at_frame)
    logger.info(
        "[BallEvents] Frame=%s | ID_type=%s | identified=%s | target_visible=%s | "
        "ball_speed=%s | dist=%s | possession=%s | event=%s | lock_confidence=%s",
        frame,
        id_type or "-",
        identified,
        sample.target_visible,
        f"{ball_speed_m_s:.1f} m/s" if ball_speed_m_s is not None else "-",
        f"{dist_m:.2f}m" if dist_m is not None else "-",
        possessed,
        event_kind or "-",
        lock_conf or "-",
    )
