"""ReID lock logic in TrackIdentityMatcher (no model load)."""

from __future__ import annotations

import numpy as np

from backend.app.pipeline.matcher import TrackIdentityMatcher


def test_reid_lock_after_prototype(monkeypatch):
    monkeypatch.setenv("LOCK_CONSECUTIVE_FRAMES", "1")
    matcher = TrackIdentityMatcher(target_jersey=9, target_name="SMITH")
    proto = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    other = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    target_embs = [proto, proto, proto]
    matcher.reid_prototype = proto
    matcher.has_number_match[7] = True
    matcher._number_streak[7] = 1
    for emb in target_embs:
        matcher.observe_reid(7, emb)
    for emb in [other, other, other]:
        matcher.observe_reid(3, emb)
    lock = matcher._try_lock_reid()
    assert lock is not None
    assert lock.track_id == 7
    assert lock.method == "reid"


def test_acquire_lock_prefers_reid_over_color(monkeypatch):
    monkeypatch.setenv("LOCK_CONSECUTIVE_FRAMES", "1")
    matcher = TrackIdentityMatcher(target_jersey=9, target_name="")
    proto = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    matcher.reid_prototype = proto
    matcher.has_number_match[7] = True
    matcher._number_streak[7] = 1
    matcher.observe_reid(7, proto)
    matcher.observe_reid(7, proto)
    for _ in range(5):
        matcher.observe_color(3, 0.9)
    lock = matcher.try_acquire_lock()
    assert lock is not None
    assert lock.method == "reid"
    assert lock.track_id == 7


def test_maybe_upgrade_color_lock_to_number(monkeypatch):
    from backend.app.pipeline.matcher import TargetLock

    monkeypatch.setenv("LOCK_CONSECUTIVE_FRAMES", "1")
    matcher = TrackIdentityMatcher(target_jersey=9, target_name="")
    matcher.lock = TargetLock(track_id=3, jersey=9, method="color", confidence=0.2)
    matcher.number_votes[3] = 1.5
    matcher.has_number_match[3] = True
    matcher._number_streak[3] = 1
    upgraded = matcher.maybe_upgrade_lock(3)
    assert upgraded is not None
    assert upgraded.method == "number"
    assert upgraded.track_id == 3
