"""Analyze/CLI share the same pitch calibration lookup order."""

from __future__ import annotations

from backend.app.pitch_api import calibration_keys_for_run


def test_no_explicit_key_matches_cli_default_only(monkeypatch):
    monkeypatch.setenv("PITCH_CALIBRATION_NAME", "testmatch2")
    assert calibration_keys_for_run(explicit_key=None) == ("testmatch2",)


def test_explicit_key_prepended_with_fallback(monkeypatch):
    monkeypatch.setenv("PITCH_CALIBRATION_NAME", "testmatch2")
    assert calibration_keys_for_run(explicit_key="lozano") == (
        "lozano",
        "testmatch2",
    )


def test_explicit_same_as_default_no_duplicate(monkeypatch):
    monkeypatch.setenv("PITCH_CALIBRATION_NAME", "testmatch2")
    assert calibration_keys_for_run(explicit_key="testmatch2") == ("testmatch2",)
