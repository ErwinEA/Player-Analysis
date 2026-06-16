"""TargetMemory fingerprint scoring and state transitions."""

from __future__ import annotations

import numpy as np

from backend.app.pipeline.target_memory import TargetMemory, memory_confidence_lock_thresh


def test_seed_and_promote_confidence(monkeypatch):
    monkeypatch.setenv("MEMORY_JERSEY_MIN_CONF", "0.35")
    mem = TargetMemory(jersey_number=22)
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    mem.seed_from_jersey(
        track_id=7,
        bbox=[100.0, 200.0, 160.0, 400.0],
        frame_idx=10,
        embedding=emb,
        jersey_conf=0.9,
    )
    assert mem.state == "LOCK_CANDIDATE"
    assert mem.locked_track_id == 7
    assert mem.confidence >= memory_confidence_lock_thresh()
    assert mem.appearance_embedding is not None


def test_on_miss_enters_searching(monkeypatch):
    monkeypatch.setenv("MEMORY_SEARCHING_FRAMES", "2")
    mem = TargetMemory(jersey_number=22)
    mem.state = "LOCKED"
    mem.on_miss(1)
    assert mem.state == "LOCKED"
    mem.on_miss(2)
    assert mem.state == "SEARCHING"
