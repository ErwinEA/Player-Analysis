"""RLE mask encode/decode roundtrip."""

from __future__ import annotations

import numpy as np

from backend.app.pipeline.mask_rle import decode_mask_rle, encode_mask_rle


def test_mask_rle_roundtrip_random():
    rng = np.random.default_rng(0)
    mask = rng.random((48, 64)) > 0.5
    rle = encode_mask_rle(mask)
    back = decode_mask_rle(rle)
    assert back.shape == mask.shape
    assert np.array_equal(back, mask)


def test_mask_rle_empty_counts_all_zeros():
    mask = np.zeros((10, 10), dtype=bool)
    rle = encode_mask_rle(mask)
    back = decode_mask_rle(rle)
    assert not back.any()
