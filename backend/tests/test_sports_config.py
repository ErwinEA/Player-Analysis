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
