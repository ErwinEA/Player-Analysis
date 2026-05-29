"""MobileSAM loader retry between analyze runs."""

from __future__ import annotations

from backend.app.pipeline import segmentation as seg


def test_prepare_mobile_sam_for_run_clears_unavailable_flag(monkeypatch):
    seg._segmenter_unavailable = True
    seg._segmenter_unavailable_reason = "test failure"
    seg.prepare_mobile_sam_for_run()
    assert seg._segmenter_unavailable is False
    assert seg._segmenter_unavailable_reason is None
