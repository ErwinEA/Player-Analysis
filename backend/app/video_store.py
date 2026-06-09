"""Default demo video on the server for calibration and analysis."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

from fastapi import HTTPException
from fastapi.responses import FileResponse

DEFAULT_VIDEO_NAME = "testmatch2"
DEFAULT_FRAME_INDEX = 100


def videos_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "videos"


def rendered_videos_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "rendered_videos"


def open_browser_video_writer(
    path: str | Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    """OpenCV writer; prefer codecs that play in HTML5 video (H.264 before mp4v)."""
    for codec in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
    raise ValueError(f"Could not open annotated video writer: {path}")


_FFMPEG_MISSING_LOGGED = False


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ensure_browser_playable_mp4(path: Path) -> bool:
    """Re-encode to H.264 when ffmpeg is available (mp4v often won't play in browsers).

    Returns True when the file exists, is non-empty, and is browser-ready (transcoded or
  already H.264). Returns False when ffmpeg is missing or transcode fails.
    """
    global _FFMPEG_MISSING_LOGGED
    if not path.is_file() or path.stat().st_size == 0:
        return False
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        if not _FFMPEG_MISSING_LOGGED:
            logger.warning(
                "ffmpeg not found; annotated MP4 may not play in browsers "
                "(install: brew install ffmpeg)"
            )
            _FFMPEG_MISSING_LOGGED = True
        return False
    tmp = path.with_suffix(".web.mp4")
    try:
        proc = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(path),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-movflags",
                "+faststart",
                "-an",
                str(tmp),
            ],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            logger.warning(
                "ffmpeg transcode failed for %s (exit %s)",
                path,
                proc.returncode,
            )
            return False
        tmp.replace(path)
        return True
    except OSError as exc:
        logger.warning("ffmpeg transcode failed for %s: %s", path, exc)
        tmp.unlink(missing_ok=True)
        return False


def rendered_video_ready(path: Path | None) -> tuple[bool, str | None]:
    """Whether an annotated analyze MP4 is safe to expose via video_url."""
    if path is None or not path.is_file():
        return False, "annotated video was not created"
    if path.stat().st_size == 0:
        return False, "annotated video file is empty"
    if not ffmpeg_available():
        return False, (
            "Install ffmpeg for browser-playable annotated video "
            "(brew install ffmpeg), then re-run Analyze"
        )
    if not ensure_browser_playable_mp4(path):
        return False, "annotated video transcode failed; check backend logs"
    return True, None


RENDERED_VIDEO_MAX_AGE_HOURS = 5.0


def rendered_video_max_age_seconds() -> float:
    """TTL for annotated analyze outputs (default 5 hours)."""
    raw = os.environ.get(
        "RENDERED_VIDEO_MAX_AGE_HOURS", str(RENDERED_VIDEO_MAX_AGE_HOURS)
    ).strip()
    try:
        hours = float(raw)
    except ValueError:
        hours = RENDERED_VIDEO_MAX_AGE_HOURS
    return max(0.0, hours) * 3600.0


def cleanup_rendered_videos(
    *,
    max_age_seconds: float | None = None,
    now: float | None = None,
) -> int:
    """Delete rendered MP4s older than max_age_seconds. Returns files removed."""
    age = rendered_video_max_age_seconds() if max_age_seconds is None else max_age_seconds
    if age <= 0:
        return 0
    out_dir = rendered_videos_dir()
    if not out_dir.is_dir():
        return 0
    cutoff = (now if now is not None else time.time()) - age
    removed = 0
    for path in out_dir.glob("*.mp4"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError as exc:
            logger.warning("Could not remove stale rendered video %s: %s", path, exc)
    if removed:
        logger.info(
            "Removed %s rendered video(s) older than %.1f hour(s)",
            removed,
            age / 3600.0,
        )
    return removed


def rendered_video_path(job_id: str) -> Path:
    safe = Path(job_id).name
    if not safe.endswith(".mp4") or safe != job_id:
        raise ValueError("Invalid rendered video id")
    return rendered_videos_dir() / safe


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


def rendered_video_response(filename: str) -> FileResponse:
    try:
        path = rendered_video_path(filename)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Video not found") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=path.name,
    )
