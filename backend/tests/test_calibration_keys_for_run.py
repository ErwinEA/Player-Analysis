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


def test_upload_filename_uses_stem_when_json_exists(monkeypatch, tmp_path):
    monkeypatch.setenv("PITCH_CALIBRATION_NAME", "testmatch2")
    monkeypatch.setattr(
        "backend.app.pitch_api.default_calibration_dir",
        lambda: tmp_path,
    )
    (tmp_path / "videoplayback.json").write_text("{}", encoding="utf-8")
    assert calibration_keys_for_run(
        explicit_key=None,
        upload_filename="videoplayback.mp4",
    ) == ("videoplayback", "testmatch2")


def test_upload_filename_ignored_without_json(monkeypatch, tmp_path):
    monkeypatch.setenv("PITCH_CALIBRATION_NAME", "testmatch2")
    monkeypatch.setattr(
        "backend.app.pitch_api.default_calibration_dir",
        lambda: tmp_path,
    )
    assert calibration_keys_for_run(
        explicit_key=None,
        upload_filename="missing_video.mp4",
    ) == ("testmatch2",)
