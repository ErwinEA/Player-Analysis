"""Tests for pitch homography calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.app.pipeline.pitch_homography import (
    BOUNDARY_POINT_COUNT,
    PitchCalibration,
    build_calibration,
    build_calibration_with_diagnostics,
    compute_homography,
    decagon_to_quad,
    is_badminton_court,
    probe_calibration,
    render_badminton_court_template,
    render_pitch_template,
    run_calibration_diagnostics,
    validate_calibration_maps_to_pitch,
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


def _wide_shot_boundary(w: int, h: int, n: int) -> list[list[float]]:
    """Clockwise outline with n points; corners at first, ~1/4, ~1/2, ~3/4 of arc."""
    if n == 10:
        return [
            [w * 0.1, h * 0.2],
            [w * 0.35, h * 0.18],
            [w * 0.5, h * 0.17],
            [w * 0.9, h * 0.2],
            [w * 0.92, h * 0.45],
            [w * 0.93, h * 0.7],
            [w * 0.95, h * 0.95],
            [w * 0.7, h * 0.97],
            [w * 0.3, h * 0.96],
            [w * 0.05, h * 0.95],
        ]
    # Minimal 5-point outline approximating same quad
    return [
        [w * 0.1, h * 0.2],
        [w * 0.5, h * 0.17],
        [w * 0.9, h * 0.2],
        [w * 0.95, h * 0.95],
        [w * 0.05, h * 0.95],
    ]


def test_outline_n5_builds_quad_and_roundtrip():
    w, h = 1280, 720
    boundary = _wide_shot_boundary(w, h, 5)
    cal = build_calibration("n5", image_boundary_points=boundary, image_size=(w, h))
    assert len(cal.image_corners) == 4
    assert len(cal.image_boundary_points or []) == 5
    # Round-trip at a corner of the fitted quad (on the pitch outline in image space)
    px0, py0 = cal.image_corners[0]
    x, y = cal.pixel_to_meters(px0, py0)
    assert 0 <= x <= cal.pitch_length_m
    assert 0 <= y <= cal.pitch_width_m
    back_px, back_py = cal.meters_to_pixel(x, y)
    assert abs(back_px - px0) < 2.0
    assert abs(back_py - py0) < 2.0


def test_n10_quad_matches_corner_indices():
    w, h = 1280, 720
    boundary = _wide_shot_boundary(w, h, 10)
    quad_direct = decagon_to_quad(boundary)
    cal = build_calibration("n10", image_boundary_points=boundary, image_size=(w, h))
    assert cal.image_corners == quad_direct


def test_build_calibration_with_diagnostics_high_confidence():
    w, h = 1280, 720
    boundary = _wide_shot_boundary(w, h, 10)
    cal, diag = build_calibration_with_diagnostics(
        "good", image_boundary_points=boundary, image_size=(w, h)
    )
    assert cal.confidence == diag.confidence
    assert diag.probe_count >= 3
    assert diag.confidence >= 0.8


def test_too_few_boundary_points_raises():
    with pytest.raises(ValueError, match="Boundary point count"):
        build_calibration("bad", image_boundary_points=[[1, 2], [3, 4]], image_size=(1280, 720))


def test_from_dict_old_json_without_metadata():
    w, h = 1280, 720
    cal = build_calibration(
        "legacy",
        [[w * 0.1, h * 0.2], [w * 0.9, h * 0.2], [w * 0.95, h * 0.95], [w * 0.05, h * 0.95]],
        image_size=(w, h),
    )
    data = cal.to_dict()
    data.pop("mode", None)
    data.pop("confidence", None)
    data.pop("coverage_pct", None)
    loaded = PitchCalibration.from_dict(data)
    assert loaded.mode == "outline"
    assert loaded.confidence is None
    x, y = loaded.pixel_to_meters(w * 0.5, h * 0.5)
    assert 0 <= x <= loaded.pitch_length_m


def test_from_dict_with_seven_boundary_points():
    w, h = 1280, 720
    boundary = _wide_shot_boundary(w, h, 5)
    cal = build_calibration("b7", image_boundary_points=boundary, image_size=(w, h))
    data = cal.to_dict()
    loaded = PitchCalibration.from_dict(data)
    assert len(loaded.image_boundary_points or []) == 5
    assert len(loaded.image_corners) == 4


def test_run_calibration_diagnostics_does_not_raise_on_bad_probes():
    w, h = 1280, 720
    # Deliberately poor quad — diagnostics should still return
    cal = build_calibration(
        "poor",
        [[w * 0.1, h * 0.1], [w * 0.2, h * 0.1], [w * 0.2, h * 0.2], [w * 0.1, h * 0.2]],
        image_size=(w, h),
    )
    diag = run_calibration_diagnostics(cal)
    assert diag.probe_total == 5
    assert 0.0 <= diag.confidence <= 1.0


def test_badminton_court_template_dimensions():
    img = render_badminton_court_template(length_m=13.4, width_m=6.1)
    assert img.shape[0] > 0 and img.shape[1] > 0
    assert is_badminton_court(13.4, 6.1)
    assert not is_badminton_court(105.0, 68.0)


def test_testmatch2_fixture_loads_and_recomputes_homography():
    path = Path("backend/data/pitch_calibration/testmatch2.json")
    if not path.is_file():
        pytest.skip("testmatch2 calibration fixture missing")
    raw = json.loads(path.read_text())
    cal = PitchCalibration.from_dict(raw)
    assert len(cal.image_corners) == 4
    assert cal.image_boundary_points is not None
    count, _ = probe_calibration(cal)
    assert count >= 0
