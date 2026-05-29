"""Tests for person bbox validation."""

from __future__ import annotations

from bbox_utils import is_valid_person_bbox, normalize_bbox


def test_rejects_giant_offscreen_bbox():
    # track 341 from testmatch3 frame 1194
    bbox = [-68.22, 147.78, 682.67, 1907.54]
    shape = (1920, 1080, 3)
    assert not is_valid_person_bbox(bbox, shape)


def test_accepts_normal_player_bbox():
    bbox = [320.75, 568.43, 793.95, 1920.0]
    shape = (1920, 1080, 3)
    assert is_valid_person_bbox(bbox, shape)


def test_normalize_clips_to_frame():
    bbox = [-10.0, 100.0, 1100.0, 2000.0]
    shape = (1920, 1080, 3)
    out = normalize_bbox(bbox, shape)
    assert out[0] >= 0
    assert out[2] <= 1080
    assert out[3] <= 1920
