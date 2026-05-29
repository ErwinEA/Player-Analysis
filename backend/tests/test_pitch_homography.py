"""Tests for pitch homography calibration."""

from __future__ import annotations

import numpy as np

from backend.app.pipeline.pitch_homography import (
    BOUNDARY_POINT_COUNT,
    build_calibration,
    compute_homography,
    decagon_to_quad,
    render_pitch_template,
)


def test_homography_roundtrip():
    corners = [[100, 200], [900, 220], [950, 650], [80, 630]]
    H, H_inv = compute_homography(corners, width=1280, height=720)
    pt = np.array([500.0, 400.0, 1.0])
    meters = H @ pt
    meters /= meters[2]
    back = H_inv @ meters
    back /= back[2]
    assert abs(back[0] - pt[0]) < 1.0
    assert abs(back[1] - pt[1]) < 1.0


def test_calibration_pixel_to_meters():
    cal = build_calibration(
        "test",
        [[115, 425], [905, 495], [1265, 705], [35, 705]],
        image_size=(1280, 720),
    )
    x, y = cal.pixel_to_meters(640, 500)
    assert 0 <= x <= cal.pitch_length_m
    assert 0 <= y <= cal.pitch_width_m
    px, py = cal.meters_to_pixel(x, y)
    assert abs(px - 640) < 2.0
    assert abs(py - 500) < 2.0


def test_render_pitch_template_shape():
    img = render_pitch_template()
    assert img.shape == (680, 1050, 3)


def test_decagon_to_quad_uses_labeled_corner_indices():
    boundary = [
        [100, 200],
        [300, 180],
        [500, 190],
        [900, 220],
        [950, 400],
        [920, 550],
        [950, 650],
        [500, 680],
        [200, 670],
        [80, 630],
    ]
    quad = decagon_to_quad(boundary)
    assert quad[0] == [100.0, 200.0]
    assert quad[1] == [900.0, 220.0]
    assert quad[2] == [950.0, 650.0]
    assert quad[3] == [80.0, 630.0]


def test_decagon_to_quad_produces_four_corners():
    boundary = [
        [100, 200],
        [300, 180],
        [500, 190],
        [900, 220],
        [950, 400],
        [920, 550],
        [950, 650],
        [500, 680],
        [200, 670],
        [80, 630],
    ]
    quad = decagon_to_quad(boundary)
    assert len(quad) == 4
    cal = build_calibration(
        "decagon",
        image_boundary_points=boundary,
        image_size=(1280, 720),
    )
    assert len(cal.image_corners) == 4
    assert cal.image_boundary_points is not None
    assert len(cal.image_boundary_points) == BOUNDARY_POINT_COUNT
    x, y = cal.pixel_to_meters(640, 500)
    assert 0 <= x <= cal.pitch_length_m
    assert 0 <= y <= cal.pitch_width_m


def test_skewed_broadcast_corners():
    """Perspective-skewed quad that previously failed convex winding check."""
    corners = [[80, 380], [980, 520], [1280, 720], [20, 700]]
    cal = build_calibration("skew", corners, image_size=(1280, 720))
    x, y = cal.pixel_to_meters(640, 550)
    assert 0 <= x <= cal.pitch_length_m
    assert 0 <= y <= cal.pitch_width_m
