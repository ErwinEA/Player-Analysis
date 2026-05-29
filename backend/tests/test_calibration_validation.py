"""Pitch calibration sanity checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.pipeline.movement_stats import compute_movement_stats_from_rows
from backend.app.pipeline.pitch_homography import (
    PitchCalibration,
    build_calibration,
    validate_calibration_maps_to_pitch,
)
from backend.app.schemas import Row


def test_validate_calibration_maps_to_pitch_accepts_good_quad():
    w, h = 1280, 720
    cal = build_calibration(
        "good",
        [
            [w * 0.1, h * 0.2],
            [w * 0.9, h * 0.2],
            [w * 0.95, h * 0.95],
            [w * 0.05, h * 0.95],
        ],
        image_size=(w, h),
    )
    validate_calibration_maps_to_pitch(cal)


def test_validate_calibration_maps_to_pitch_rejects_testmatch2_file():
    path = Path("backend/data/pitch_calibration/testmatch2.json")
    if not path.is_file():
        pytest.skip("testmatch2 calibration fixture missing")
    cal = PitchCalibration.from_dict(json.loads(path.read_text()))
    with pytest.raises(ValueError, match="does not map"):
        validate_calibration_maps_to_pitch(cal)


def test_movement_stats_match_heatmap_bounds_filtering():
    cal = PitchCalibration.from_dict(
        json.loads(
            Path("backend/data/pitch_calibration/testmatch2.json").read_text()
        )
    )
    rows = [
        Row(
            t=i / 30.0,
            frame=i,
            track=1,
            player=10,
            x=0.5,
            y=0.5,
            bbox=[200.0 + i * 40, 500.0, 250.0 + i * 40, 600.0],
        )
        for i in range(20)
    ]
    stats = compute_movement_stats_from_rows(rows, fps=30.0, calibration=cal)
    assert stats.distance_m == 0.0
    assert stats.units == "meters"
