"""Temporal jersey-match streak before number/ReID lock."""

from __future__ import annotations

import numpy as np

from backend.app.pipeline.matcher import TargetLock, TrackIdentityMatcher


def _observe_target(
    matcher: TrackIdentityMatcher,
    track_id: int,
    frame_idx: int,
    conf: float = 0.9,
    *,
    ocr_every: int = 1,
) -> None:
    matcher.begin_frame(frame_idx, ocr_every=ocr_every)
    matcher.observe_number(
        track_id,
        matcher.target_jersey,
        conf,
        frame_idx=frame_idx,
        ocr_every=ocr_every,
    )


def test_single_frame_does_not_lock_by_number(monkeypatch):
    monkeypatch.setenv("NUMBER_LOCK_VOTES", "0.5")
    monkeypatch.setenv("LOCK_CONSECUTIVE_FRAMES", "2")
    matcher = TrackIdentityMatcher(target_jersey=9, target_name="")
    _observe_target(matcher, 7, 10, conf=1.0)
    assert matcher.try_acquire_lock() is None


def test_two_consecutive_frames_lock_by_number(monkeypatch):
    monkeypatch.setenv("NUMBER_LOCK_VOTES", "0.5")
    monkeypatch.setenv("LOCK_CONSECUTIVE_FRAMES", "2")
    matcher = TrackIdentityMatcher(target_jersey=9, target_name="")
    _observe_target(matcher, 7, 10)
    _observe_target(matcher, 7, 11)
    lock = matcher.try_acquire_lock()
    assert lock is not None
    assert lock.method == "number"
    assert lock.track_id == 7


def test_frame_gap_resets_streak(monkeypatch):
    monkeypatch.setenv("NUMBER_LOCK_VOTES", "0.5")
    monkeypatch.setenv("LOCK_CONSECUTIVE_FRAMES", "2")
    matcher = TrackIdentityMatcher(target_jersey=9, target_name="")
    _observe_target(matcher, 7, 10)
    _observe_target(matcher, 7, 12)
    assert matcher.try_acquire_lock() is None


def test_other_track_vote_does_not_reset_streak(monkeypatch):
    monkeypatch.setenv("NUMBER_LOCK_VOTES", "0.5")
    monkeypatch.setenv("LOCK_CONSECUTIVE_FRAMES", "2")
    matcher = TrackIdentityMatcher(target_jersey=9, target_name="")
    matcher.begin_frame(10, ocr_every=1)
    matcher.observe_number(7, 9, 0.9, frame_idx=10, ocr_every=1)
    matcher.observe_number(3, 9, 0.9, frame_idx=10, ocr_every=1)
    _observe_target(matcher, 7, 11, ocr_every=1)
    lock = matcher.try_acquire_lock()
    assert lock is not None
    assert lock.track_id == 7


def test_streak_survives_ocr_every_spacing(monkeypatch):
    monkeypatch.setenv("NUMBER_LOCK_VOTES", "0.5")
    monkeypatch.setenv("LOCK_CONSECUTIVE_FRAMES", "2")
    matcher = TrackIdentityMatcher(target_jersey=9, target_name="")
    _observe_target(matcher, 7, 0, ocr_every=3)
    _observe_target(matcher, 7, 3, ocr_every=3)
    lock = matcher.try_acquire_lock()
    assert lock is not None
    assert lock.method == "number"
    assert lock.track_id == 7


def test_release_lock_clears_streak(monkeypatch):
    monkeypatch.setenv("NUMBER_LOCK_VOTES", "0.5")
    monkeypatch.setenv("LOCK_CONSECUTIVE_FRAMES", "2")
    matcher = TrackIdentityMatcher(target_jersey=9, target_name="")
    _observe_target(matcher, 7, 10)
    _observe_target(matcher, 7, 11)
    matcher.release_lock()
    matcher.number_votes[7] = 2.0
    matcher.has_number_match[7] = True
    assert matcher.try_acquire_lock() is None


def test_color_lock_ignores_temporal_streak():
    matcher = TrackIdentityMatcher(target_jersey=9, target_name="")
    for _ in range(5):
        matcher.observe_color(3, 0.9)
    lock = matcher.try_acquire_lock()
    assert lock is not None
    assert lock.method == "color"


def test_reid_lock_requires_temporal_streak(monkeypatch):
    monkeypatch.setenv("LOCK_CONSECUTIVE_FRAMES", "2")
    matcher = TrackIdentityMatcher(target_jersey=9, target_name="")
    proto = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    matcher.reid_prototype = proto
    matcher.begin_frame(0, ocr_every=1)
    matcher.observe_number(7, 9, 0.9, frame_idx=0, ocr_every=1)
    matcher.observe_reid(7, proto)
    matcher.observe_reid(7, proto)
    assert matcher._try_lock_reid() is None
    matcher.begin_frame(1, ocr_every=1)
    matcher.observe_number(7, 9, 0.9, frame_idx=1, ocr_every=1)
    lock = matcher._try_lock_reid()
    assert lock is not None
    assert lock.method == "reid"


def test_maybe_upgrade_requires_temporal_streak(monkeypatch):
    monkeypatch.setenv("NUMBER_LOCK_VOTES", "0.5")
    monkeypatch.setenv("LOCK_CONSECUTIVE_FRAMES", "2")
    matcher = TrackIdentityMatcher(target_jersey=9, target_name="")
    matcher.lock = TargetLock(track_id=3, jersey=9, method="color", confidence=0.2)
    matcher.number_votes[3] = 1.5
    matcher.has_number_match[3] = True
    assert matcher.maybe_upgrade_lock(3) is not None
    assert matcher.lock.method == "color"
    _observe_target(matcher, 3, 10)
    _observe_target(matcher, 3, 11)
    upgraded = matcher.maybe_upgrade_lock(3)
    assert upgraded is not None
    assert upgraded.method == "number"
