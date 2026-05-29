from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.app.pitch_api import (
    calibration_key_from_filename,
    get_pitch_calibration,
    get_pitch_frame,
    legacy_calibration_video_dest,
    legacy_calibration_video_stored,
    normalize_calibration_name,
    pitch_template_response,
    save_pitch_calibration,
    save_pitch_calibration_upload,
)
from backend.app.pipeline.pitch_homography import default_calibration_dir
from backend.app.pipeline.run import run_pipeline
from backend.app.pipeline.segmentation import mobile_sam_health
from backend.app.schemas import (
    AnalyzeResponse,
    PitchCalibrationSaveRequest,
    PitchCalibrationSaveResponse,
    PitchFrameResponse,
    PlayerDetails,
)
from backend.app.upload_utils import max_upload_bytes, stream_upload_capped
from backend.app.video_store import default_video_response, get_default_video_file

logger = logging.getLogger(__name__)

_DEFAULT_CORS = "http://localhost:3000,http://127.0.0.1:3000"


def _cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", _DEFAULT_CORS)
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(title="Player Analysis Tracking API", version="0.1.0")

_cors_kwargs: dict = {
    "allow_origins": _cors_origins(),
    "allow_credentials": False,
    "allow_methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["*"],
}
if os.environ.get("CORS_ALLOW_LOCAL_REGEX", "1").strip() not in ("0", "false", "no"):
    _cors_kwargs["allow_origin_regex"] = r"http://(localhost|127\.0\.0\.1)(:\d+)?"

app.add_middleware(CORSMiddleware, **_cors_kwargs)
@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "Player Analysis Tracking API",
        "health": "/health",
        "analyze": "POST /api/analyze",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, object]:
    sam = mobile_sam_health()
    top_status = "error" if sam.get("status") == "error" else "ok"
    return {"status": top_status, "mobile_sam": sam}


@app.get("/api/pitch/template")
def pitch_template(name: str = "testmatch2"):
    return pitch_template_response(name)


@app.get("/api/pitch/calibration")
def pitch_calibration(name: str = "testmatch2") -> dict:
    return get_pitch_calibration(name)


@app.get("/api/videos/default")
def serve_default_video():
    return default_video_response()


@app.get("/api/videos/default/info")
def default_video_info() -> dict:
    try:
        path = get_default_video_file()
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "name": path.stem,
        "filename": path.name,
        "url": "/api/videos/default",
        "frame_url": "/api/pitch/frame",
        "default_frame_index": 100,
    }


@app.get("/api/pitch/frame", response_model=PitchFrameResponse)
def pitch_frame(
    name: str = "testmatch2",
    frame: int = 100,
) -> PitchFrameResponse:
    return get_pitch_frame(name=name, frame_index=frame)


@app.get("/api/pitch/video/stored")
def pitch_video_stored(name: str) -> dict[str, bool | str]:
    """Whether a legacy calibration video is already on disk for this name."""
    try:
        safe = normalize_calibration_name(name)
    except HTTPException:
        return {"name": name, "stored": False}
    return {"name": safe, "stored": legacy_calibration_video_stored(safe)}


@app.post("/api/pitch/calibration", response_model=PitchCalibrationSaveResponse)
def pitch_calibration_save(
    body: PitchCalibrationSaveRequest,
) -> PitchCalibrationSaveResponse:
    return save_pitch_calibration(body)


@app.post(
    "/api/pitch/calibration/upload",
    response_model=PitchCalibrationSaveResponse,
)
async def pitch_calibration_upload(
    name: str = Form(...),
    frame_index: int = Form(100),
    image_corners: str = Form(...),
    video: UploadFile = File(...),
) -> PitchCalibrationSaveResponse:
    """Save pitch homography from an uploaded legacy video (stored server-side)."""
    import json

    try:
        corners = json.loads(image_corners)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="Invalid image_corners JSON") from exc
    if not isinstance(corners, list):
        raise HTTPException(status_code=422, detail="image_corners must be a JSON array")

    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    dest = legacy_calibration_video_dest(name, suffix)
    default_calibration_dir().mkdir(parents=True, exist_ok=True)
    await stream_upload_capped(
        video,
        dest,
        max_bytes=max_upload_bytes(),
    )
    try:
        kwargs: dict = {"name": name, "frame_index": frame_index, "video_path": dest}
        if len(corners) == 10:
            kwargs["image_boundary_points"] = corners
        elif len(corners) == 4:
            kwargs["image_corners"] = corners
        else:
            raise HTTPException(
                status_code=422,
                detail="Calibration points must be 10 (boundary) or 4 (legacy corners).",
            )
        return save_pitch_calibration_upload(**kwargs)
    except HTTPException:
        raise
    except OSError as exc:
        if exc.errno == 28:
            raise HTTPException(
                status_code=507,
                detail=(
                    "Not enough disk space to save calibration. "
                    "Free disk space and try again."
                ),
            ) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/analyze/default", response_model=AnalyzeResponse)
async def analyze_default_video(details: str = Form(...)) -> AnalyzeResponse:
    """Analyze the server-hosted default demo video (no upload)."""
    try:
        player_details = PlayerDetails.model_validate_json(details)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid details JSON: {exc}") from exc
    if player_details.jerseyNumber <= 0:
        raise HTTPException(status_code=422, detail="jerseyNumber must be between 1 and 99")

    video_path = str(get_default_video_file())
    return run_pipeline(
        video_path,
        player_details,
        calibration_key="testmatch2",
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    video: UploadFile = File(...),
    details: str = Form(...),
) -> AnalyzeResponse:
    try:
        player_details = PlayerDetails.model_validate_json(details)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid details JSON: {exc}") from exc
    if player_details.jerseyNumber <= 0:
        raise HTTPException(status_code=422, detail="jerseyNumber must be between 1 and 99")

    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
        written = await stream_upload_capped(
            video,
            Path(tmp_path),
            max_bytes=max_upload_bytes(),
        )
        if written == 0:
            raise HTTPException(status_code=422, detail="Uploaded video file is empty")

        calibration_key = calibration_key_from_filename(
            video.filename or "upload.mp4",
        )
        return run_pipeline(
            tmp_path,
            player_details,
            calibration_key=calibration_key,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Analyze failed")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Check server logs for details.",
        ) from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
