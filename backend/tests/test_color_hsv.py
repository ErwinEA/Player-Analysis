"""HSV jersey color scoring."""

import numpy as np

from backend.app.pipeline.color import hex_pixel_match_score_hsv, jersey_color_score


def test_hex_pixel_match_score_hsv_accepts_bgr_crop():
    crop = np.full((40, 30, 3), (255, 0, 0), dtype=np.uint8)  # blue BGR
    score = hex_pixel_match_score_hsv(crop, "#0000ff")
    assert 0.0 <= score <= 1.0
    assert score > 0.5


def test_jersey_color_score_hsv_path():
    frame = np.full((200, 100, 3), (0, 0, 255), dtype=np.uint8)
    bbox = [10.0, 20.0, 80.0, 180.0]
    score = jersey_color_score(
        frame, bbox, "#ff0000", "", use_hsv=True
    )
    assert score >= 0.0
