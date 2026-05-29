"""Tests for bbox validity helpers."""

from __future__ import annotations

from backend.app.pipeline.bbox import is_valid_bbox, is_valid_person_bbox

FRAME = (720, 1280)  # h, w


def test_is_valid_bbox_min_size():
    assert is_valid_bbox([100, 100, 140, 140], FRAME, min_size=16)
    assert not is_valid_bbox([100, 100, 108, 108], FRAME, min_size=16)


def test_person_bbox_accepts_typical_player():
    # Upright player: taller than wide, reasonable size.
    assert is_valid_person_bbox([500, 300, 560, 460], FRAME)


def test_person_bbox_rejects_wide_graphic():
    # Scoreboard / banner: much wider than tall.
    assert not is_valid_person_bbox([100, 50, 700, 110], FRAME)


def test_person_bbox_rejects_tiny_sliver():
    assert not is_valid_person_bbox([10, 10, 30, 200], FRAME)


def test_person_bbox_rejects_full_frame_box():
    # Off-screen giant covering most of the frame.
    assert not is_valid_person_bbox([0, 0, 1280, 720], FRAME)
