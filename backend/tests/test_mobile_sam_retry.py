"""MobileSAM load retry cooldown."""

from __future__ import annotations

import time

from backend.app.pipeline import segmentation as seg


def test_unavailable_retries_after_cooldown(monkeypatch):
    monkeypatch.setenv("SAM_ENABLED", "1")
    seg._segmenter = None
    seg._segmenter_unavailable = True
    seg._segmenter_unavailable_reason = "test"
    seg._segmenter_load_attempted_at = time.time() - 60.0

    # Will fail on weights/load but must not return None solely due to unavailable flag
    seg.get_mobile_sam_segmenter()
    assert seg._segmenter_load_attempted_at > time.time() - 5.0
