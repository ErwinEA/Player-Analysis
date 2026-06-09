"""Tests for annotated analyze video helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from backend.app import main as main_app
from backend.app import video_store
from backend.app.schemas import AnalyzeResponse, TargetMatch, VideoMeta


def _minimal_analyze_response() -> AnalyzeResponse:
    return AnalyzeResponse(
        video=VideoMeta(
            fps=30.0,
            width=1280,
            height=720,
            frames=100,
            duration_s=3.33,
        ),
        target=TargetMatch(jersey=10, track_id=1, confidence=0.9, method="number"),
        rows=[],
    )


def test_parse_render_video_flag() -> None:
    assert main_app._parse_render_video_flag(None) is False
    assert main_app._parse_render_video_flag("false") is False
    assert main_app._parse_render_video_flag("true") is True
    assert main_app._parse_render_video_flag("1") is True
    assert main_app._parse_render_video_flag("yes") is True


def test_rendered_video_path_rejects_traversal() -> None:
    with pytest.raises(ValueError):
        video_store.rendered_video_path("../evil.mp4")


def test_rendered_video_ready_rejects_empty_file(tmp_path: Path) -> None:
    clip = tmp_path / "empty.mp4"
    clip.write_bytes(b"")
    ok, reason = video_store.rendered_video_ready(clip)
    assert ok is False
    assert reason is not None
    assert "empty" in reason.lower()


def test_rendered_video_ready_requires_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not-empty")
    monkeypatch.setattr(video_store.shutil, "which", lambda _: None)
    ok, reason = video_store.rendered_video_ready(clip)
    assert ok is False
    assert reason is not None
    assert "ffmpeg" in reason.lower()


def test_finalize_analyze_response_attaches_url_when_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip = tmp_path / "ok.mp4"
    clip.write_bytes(b"clip")
    monkeypatch.setattr(
        main_app, "rendered_video_ready", lambda _p: (True, None)
    )
    result = main_app._finalize_analyze_response(
        _minimal_analyze_response(),
        render_requested=True,
        rendered_path=clip,
        video_url="/api/videos/rendered/abc.mp4",
    )
    assert result.video_url == "/api/videos/rendered/abc.mp4"
    assert result.video_unavailable_reason is None
    assert clip.is_file()


def test_finalize_analyze_response_surfaces_reason_and_unlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip = tmp_path / "bad.mp4"
    clip.write_bytes(b"clip")
    monkeypatch.setattr(
        main_app,
        "rendered_video_ready",
        lambda _p: (False, "transcode failed"),
    )
    result = main_app._finalize_analyze_response(
        _minimal_analyze_response(),
        render_requested=True,
        rendered_path=clip,
        video_url="/api/videos/rendered/bad.mp4",
    )
    assert result.video_url is None
    assert result.video_unavailable_reason == "transcode failed"
    assert not clip.exists()


def test_video_meta_accepts_frame_start() -> None:
    meta = VideoMeta(
        fps=30.0,
        width=100,
        height=100,
        frames=10,
        duration_s=1.0,
        frame_start=970,
    )
    assert meta.frame_start == 970


def test_video_meta_frame_start_defaults_to_zero() -> None:
    meta = VideoMeta(
        fps=30.0,
        width=100,
        height=100,
        frames=10,
        duration_s=1.0,
    )
    assert meta.frame_start == 0
