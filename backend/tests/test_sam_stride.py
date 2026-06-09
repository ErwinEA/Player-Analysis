"""SAM_STRIDE gating (no torch)."""

from backend.app.pipeline.segmentation import (
    sam_stride,
    segment_result_from_bbox,
    should_run_sam_this_frame,
)


def test_sam_stride_default(monkeypatch):
    monkeypatch.delenv("SAM_STRIDE", raising=False)
    assert sam_stride() == 1
    assert should_run_sam_this_frame(10, None)
    assert should_run_sam_this_frame(10, 9)


def test_sam_stride_three(monkeypatch):
    monkeypatch.setenv("SAM_STRIDE", "3")
    assert sam_stride() == 3
    assert should_run_sam_this_frame(100, None)
    assert should_run_sam_this_frame(101, 100) is False
    assert should_run_sam_this_frame(103, 100) is True
    assert should_run_sam_this_frame(105, 103) is False
    assert should_run_sam_this_frame(106, 103) is True


def test_segment_result_from_bbox_foot():
    res = segment_result_from_bbox([100.0, 50.0, 200.0, 250.0], 7, 1920, 1080)
    assert res.segment_track_id == 7
    assert res.mask_rle is None
    assert res.segment_source == "bbox_fallback"
    assert res.foot_nx == round(150.0 / 1920, 4)
    assert res.foot_ny == round(250.0 / 1080, 4)
