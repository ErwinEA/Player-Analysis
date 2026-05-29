"""Shared upload size limits and streaming writes."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException, UploadFile

_READ_CHUNK = 1024 * 1024


def max_upload_bytes() -> int:
    try:
        mb = int(os.environ.get("MAX_UPLOAD_MB", "4096"))
    except ValueError:
        mb = 4096
    return max(1, mb) * 1024 * 1024


async def read_upload_capped(video: UploadFile, *, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await video.read(_READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Video exceeds upload limit ({max_bytes // (1024 * 1024)} MB).",
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def stream_upload_capped(
    video: UploadFile,
    dest: Path,
    *,
    max_bytes: int,
) -> int:
    """Stream upload to disk; returns bytes written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await video.read(_READ_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Video exceeds upload limit ({max_bytes // (1024 * 1024)} MB)."
                        ),
                    )
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except OSError as exc:
        dest.unlink(missing_ok=True)
        if exc.errno == 28:
            raise HTTPException(
                status_code=507,
                detail=(
                    "Not enough disk space to store the video. "
                    "Free disk space and try again."
                ),
            ) from exc
        raise HTTPException(
            status_code=500,
            detail=f"Could not save video: {exc}",
        ) from exc
    return total
