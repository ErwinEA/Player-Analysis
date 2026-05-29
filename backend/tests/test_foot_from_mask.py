"""Foot position from segmentation mask."""

from __future__ import annotations

import numpy as np

from backend.app.pipeline.segmentation import foot_from_mask


def test_foot_from_mask_lowest_row_median_x():
    mask = np.zeros((100, 80), dtype=bool)
    mask[90:100, 30:50] = True
    mask[95:100, 60:70] = True
    foot = foot_from_mask(mask)
    assert foot is not None
    x, y = foot
    assert y == 99.0
    assert 30 <= x <= 70


def test_foot_from_mask_empty_returns_none():
    assert foot_from_mask(np.zeros((10, 10), dtype=bool)) is None
