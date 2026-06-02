"""ReID lock logic in TrackIdentityMatcher (no model load)."""

from __future__ import annotations

import numpy as np

from backend.app.pipeline.matcher import TrackIdentityMatcher


def test_reid_lock_after_prototype():
    matcher = TrackIdentityMatcher(target_jersey=9, target_name="SMITH")
    proto = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    other = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    target_embs = [proto, proto, proto]
    matcher.reid_prototype = proto
    matcher.has_number_match[7] = True
    for emb in target_embs:
        matcher.observe_reid(7, emb)
    for emb in [other, other, other]:
        matcher.observe_reid(3, emb)
    lock = matcher._try_lock_reid()
    assert lock is not None
    assert lock.track_id == 7
    assert lock.method == "reid"
