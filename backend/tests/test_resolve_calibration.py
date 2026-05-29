"""Tests for safe pitch calibration resolution."""

from __future__ import annotations

from backend.app.pitch_api import resolve_calibration


def test_resolve_calibration_prefers_upload_key(tmp_path, monkeypatch):
    import backend.app.pitch_api as pitch_api_mod

    cal_dir = tmp_path / "cal"
    cal_dir.mkdir()
    (cal_dir / "testmatch3.json").write_text(
        '{"name":"testmatch3","image_size":[1280,720],'
        '"image_corners":[[115,425],[905,495],[1265,705],[35,705]]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(pitch_api_mod, "default_calibration_dir", lambda: cal_dir)

    cal, name = resolve_calibration("testmatch3", "tmpRandomStem", "testmatch2")
    assert name == "testmatch3"
    assert cal is not None
    assert cal.name == "testmatch3"


def test_resolve_calibration_rejects_unsafe_names(tmp_path, monkeypatch):
    import backend.app.pitch_api as pitch_api_mod

    cal_dir = tmp_path / "cal"
    cal_dir.mkdir()
    monkeypatch.setattr(pitch_api_mod, "default_calibration_dir", lambda: cal_dir)

    cal, name = resolve_calibration("../../etc/passwd", "also-bad/../x")
    assert cal is None
    assert name is None
