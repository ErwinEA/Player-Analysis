"""Tests for rendered analyze video TTL cleanup."""

from __future__ import annotations

import time
from pathlib import Path

from backend.app import video_store


def test_cleanup_rendered_videos_removes_stale_files(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(video_store, "rendered_videos_dir", lambda: tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    stale = tmp_path / "stale.mp4"
    fresh = tmp_path / "fresh.mp4"
    stale.write_bytes(b"old")
    fresh.write_bytes(b"new")
    now = time.time()
    stale.touch()
    fresh.touch()
    import os

    os.utime(stale, (now - 6 * 3600, now - 6 * 3600))
    os.utime(fresh, (now - 3600, now - 3600))

    removed = video_store.cleanup_rendered_videos(
        max_age_seconds=5 * 3600,
        now=now,
    )

    assert removed == 1
    assert not stale.exists()
    assert fresh.is_file()


def test_cleanup_rendered_videos_keeps_recent_files(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(video_store, "rendered_videos_dir", lambda: tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    clip = tmp_path / "recent.mp4"
    clip.write_bytes(b"clip")
    now = time.time()
    clip.touch()
    import os

    os.utime(clip, (now - 1800, now - 1800))

    removed = video_store.cleanup_rendered_videos(
        max_age_seconds=5 * 3600,
        now=now,
    )

    assert removed == 0
    assert clip.is_file()
