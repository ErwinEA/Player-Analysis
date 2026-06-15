"""Tests for sport court dimension config."""

from backend.app.pipeline.sports.config import (
    calibration_matches_sport,
    court_dimensions,
)


def test_court_dimensions_football():
    length_m, width_m = court_dimensions("football")
    assert length_m == 105.0
    assert width_m == 68.0


def test_court_dimensions_badminton():
    length_m, width_m = court_dimensions("badminton")
    assert length_m == 13.4
    assert width_m == 6.1


def test_court_dimensions_unknown_defaults_football():
    length_m, width_m = court_dimensions("unknown")
    assert length_m == 105.0
    assert width_m == 68.0


def test_calibration_matches_sport_badminton():
    assert calibration_matches_sport(
        sport="badminton",
        pitch_length_m=13.4,
        pitch_width_m=6.1,
    )


def test_calibration_mismatch_football_on_badminton():
    assert not calibration_matches_sport(
        sport="badminton",
        pitch_length_m=105.0,
        pitch_width_m=68.0,
    )


def test_template_path_separate_for_badminton():
    from backend.app.pitch_api import _template_path

    fb = _template_path("demo", 105.0, 68.0)
    bd = _template_path("demo", 13.4, 6.1)
    assert fb.name == "demo_topview_template.png"
    assert bd.name == "demo_badminton_topview_template.png"
    assert fb != bd


def test_partial_court_dims_rejected_by_schema():
    import pytest
    from pydantic import ValidationError

    from backend.app.schemas import PitchCalibrationSaveRequest

    corners = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    with pytest.raises(ValidationError):
        PitchCalibrationSaveRequest(
            name="demo",
            frame_index=0,
            image_corners=corners,
            pitch_length_m=13.4,
        )
