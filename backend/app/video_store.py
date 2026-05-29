"""Default demo video on the server for calibration and analysis."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse

DEFAULT_VIDEO_NAME = "testmatch2"
DEFAULT_FRAME_INDEX = 100


def videos_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "videos"


def default_video_path() -> Path:
    raw = os.environ.get("DEFAULT_VIDEO_PATH", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return videos_dir() / f"{DEFAULT_VIDEO_NAME}.mp4"


def _calibration_video_path() -> Path | None:
    cal_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "pitch_calibration"
        / f"{DEFAULT_VIDEO_NAME}.json"
    )
    if not cal_path.is_file():
        return None
    try:
        data = json.loads(cal_path.read_text(encoding="utf-8"))
        vp = data.get("video_path")
        if vp and Path(vp).expanduser().is_file():
            return Path(vp).expanduser().resolve()
    except (json.JSONDecodeError, OSError):
        pass
    return None


def ensure_default_video_installed() -> Path:
    """Copy source video into backend/data/videos if missing."""
    dest = default_video_path()
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for candidate in (
        _calibration_video_path(),
        Path(os.environ.get("DEFAULT_VIDEO_SOURCE", "")).expanduser()
        if os.environ.get("DEFAULT_VIDEO_SOURCE")
        else None,
    ):
        if candidate is not None and candidate.is_file():
            shutil.copy2(candidate, dest)
            return dest
    raise FileNotFoundError(
        f"Default video not found at {dest}. Set DEFAULT_VIDEO_PATH or "
        f"DEFAULT_VIDEO_SOURCE, or place {DEFAULT_VIDEO_NAME}.mp4 in {videos_dir()}"
    )


def get_default_video_file() -> Path:
    try:
        return ensure_default_video_installed()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def default_video_response() -> FileResponse:
    path = get_default_video_file()
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=path.name,
    )
