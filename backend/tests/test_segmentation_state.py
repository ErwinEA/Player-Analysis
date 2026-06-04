"""Segmentation warmup state (no torch)."""

from __future__ import annotations

from backend.app.pipeline.segmentation import SegmentationState, sam_warmup_frames


def test_sam_active_after_warmup(monkeypatch):
    monkeypatch.setenv("SAM_WARMUP_FRAMES", "3")
    state = SegmentationState()
    state.acquire_lock(7)
    state.record_visible_frame(10)
    state.record_visible_frame(11)
    assert not state.sam_active
    state.record_visible_frame(12)
    assert state.sam_active
    assert state.segmentation_started_at_frame == 12


def test_consecutive_resets_when_missing_before_sam(monkeypatch):
    monkeypatch.setenv("SAM_WARMUP_FRAMES", "3")
    monkeypatch.setenv("SAM_WARMUP_MISS_FRAMES", "0")
    state = SegmentationState()
    state.acquire_lock(1)
    state.record_visible_frame(1)
    state.record_visible_frame(2)
    state.record_miss_frame()
    state.record_visible_frame(3)
    assert state.visible_count == 1
    assert not state.sam_active


def test_miss_grace_preserves_warmup_progress(monkeypatch):
    monkeypatch.setenv("SAM_WARMUP_FRAMES", "3")
    monkeypatch.setenv("SAM_WARMUP_MISS_FRAMES", "15")
    state = SegmentationState()
    state.acquire_lock(1)
    state.record_visible_frame(1)
    state.record_visible_frame(2)
    state.record_miss_frame()
    state.record_visible_frame(3)
    assert state.sam_active


def test_warmup_default_is_one(monkeypatch):
    monkeypatch.delenv("SAM_WARMUP_FRAMES", raising=False)
    assert sam_warmup_frames() == 1


def test_acquire_lock_does_not_deactivate_sam_when_track_changes():
    state = SegmentationState()
    state.acquire_lock(1)
    state.record_visible_frame(10)
    assert state.sam_active
    state.acquire_lock(2)
    assert state.sam_active
    assert state.lock_track_id == 2


def test_lock_frame_counts_when_visible_same_frame_as_acquire(monkeypatch):
    """Lock frame: acquire_lock then record_visible_frame on same frame."""
    monkeypatch.setenv("SAM_WARMUP_FRAMES", "1")
    state = SegmentationState()
    state.acquire_lock(7)
    state.record_visible_frame(10)
    assert state.sam_active
    assert state.segmentation_started_at_frame == 10
