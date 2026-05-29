"""Binary mask run-length encoding (row-major / C order)."""

from __future__ import annotations

import numpy as np

from pydantic import BaseModel


class MaskRle(BaseModel):
    """Full-resolution binary mask as RLE counts (row-major flatten)."""

    size: list[int]  # [height, width]
    counts: list[int]


def encode_mask_rle(mask: np.ndarray) -> MaskRle:
    """Encode a 2D bool mask. Counts alternate from background (0) run length."""
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    h, w = mask.shape
    flat = np.asarray(mask, dtype=bool).reshape(-1)
    if flat.size == 0:
        return MaskRle(size=[int(h), int(w)], counts=[0])

    counts: list[int] = []
    cur_val = bool(flat[0])
    run = 1
    for val in flat[1:]:
        bit = bool(val)
        if bit == cur_val:
            run += 1
        else:
            counts.append(run)
            cur_val = bit
            run = 1
    counts.append(run)
    if flat[0]:
        counts.insert(0, 0)
    return MaskRle(size=[int(h), int(w)], counts=counts)


def decode_mask_rle(rle: MaskRle | dict) -> np.ndarray:
    """Decode RLE to bool mask (h, w), row-major."""
    if isinstance(rle, dict):
        rle = MaskRle.model_validate(rle)
    h, w = int(rle.size[0]), int(rle.size[1])
    total = h * w
    flat = np.zeros(total, dtype=np.uint8)
    idx = 0
    val = 0
    for run in rle.counts:
        end = min(idx + int(run), total)
        flat[idx:end] = val
        idx = end
        val = 1 - val
    return flat.reshape((h, w)).astype(bool)
