from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.app.pitch_api import (
    get_pitch_calibration,
    get_pitch_frame,
    legacy_calibration_video_dest,
    legacy_calibration_video_stored,
    normalize_calibration_name,
    pitch_template_response,
    preview_pitch_calibration,
    save_pitch_calibration,
    save_pitch_calibration_upload,
)
from backend.app.pipeline.pitch_homography import default_calibration_dir
from backend.app.pipeline.run import run_pipeline
from backend.app.pipeline.badminton.shuttle_detector import shuttle_health
from backend.app.pipeline.segmentation import mobile_sam_health
from backend.app.pipeline.insights import generate_insights, ollama_health
from backend.app.schemas import (
    AnalyzeResponse,
    InsightsRequest,
    InsightsResponse,
    PitchCalibrationPreviewResponse,
    PitchCalibrationSaveRequest,
    PitchCalibrationSaveResponse,
    PitchFrameResponse,
    PlayerDetails,
)
from backend.app.upload_utils import max_upload_bytes, stream_upload_capped
from backend.app.video_store import (
    cleanup_rendered_videos,
    default_video_response,
    get_default_video_file,
    rendered_video_ready,
    rendered_video_response,
    rendered_videos_dir,
)

logger = logging.getLogger(__name__)

_DEFAULT_CORS = "http://localhost:3000,http://127.0.0.1:3000"


def _configure_app_logging() -> None:
    """Ensure pipeline INFO logs (YOLO/SAM load, analyze progress) reach the terminal."""
    fmt = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        root.addHandler(handler)
    for name in ("backend", "backend.app", "ultralytics"):
        child = logging.getLogger(name)
        child.setLevel(logging.INFO)


def _cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", _DEFAULT_CORS)
    return [o.strip() for o in raw.split(",") if o.strip()]


def _parse_render_video_flag(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in ("true", "1", "yes", "on")


def _allocate_rendered_output() -> tuple[Path, str]:
    job_id = f"{uuid.uuid4().hex}.mp4"
    out_dir = rendered_videos_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / job_id, f"/api/videos/rendered/{job_id}"


def _unlink_rendered(path: Path | None) -> None:
    if path is not None and path.is_file():
        path.unlink(missing_ok=True)


def _finalize_analyze_response(
    result: AnalyzeResponse,
    *,
    render_requested: bool,
    rendered_path: Path | None,
    video_url: str | None,
) -> AnalyzeResponse:
    if not render_requested:
        return result
    ready, reason = (
        rendered_video_ready(rendered_path)
        if rendered_path is not None
        else (False, "annotated video was not created")
    )
    if ready and video_url:
        return result.model_copy(update={"video_url": video_url})
    _unlink_rendered(rendered_path)
    return result.model_copy(
        update={
            "video_unavailable_reason": reason
            or "annotated video unavailable",
        }
    )


@asynccontextmanager
async def _app_lifespan(_app: FastAPI):
    from backend.app.pipeline.segmentation import log_runtime_devices

    _configure_app_logging()
    from backend.app.pipeline.run import _max_frames

    log_runtime_devices()
    logger.info("MAX_FRAMES cap: %s", _max_frames())
    cleanup_rendered_videos()
    yield


app = FastAPI(
    title="Player Analysis Tracking API",
    version="0.1.0",
    lifespan=_app_lifespan,
)

_cors_kwargs: dict = {
    "allow_origins": _cors_origins(),
    "allow_credentials": False,
    "allow_methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["*"],
}
if os.environ.get("CORS_ALLOW_LOCAL_REGEX", "1").strip() not in ("0", "false", "no"):
    # localhost, loopback, and common LAN dev URLs (Next.js "Network" link)
    _cors_kwargs["allow_origin_regex"] = (
        r"http://(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3})(:\d+)?"
    )


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
    return {
        "status": top_status,
        "mobile_sam": sam,
        "shuttle": shuttle_health(),
        "ollama": ollama_health(),
    }


@app.post("/api/insights", response_model=InsightsResponse)
async def insights(body: InsightsRequest) -> InsightsResponse:
    return await asyncio.to_thread(generate_insights, body)


@app.get("/api/pitch/template")
def pitch_template(name: str = "testmatch2", sport: str | None = None):
    return pitch_template_response(name, sport=sport)


@app.get("/api/pitch/calibration")
def pitch_calibration(name: str = "testmatch2") -> dict:
    return get_pitch_calibration(name)


@app.get("/api/videos/default")
def serve_default_video():
    return default_video_response()


@app.get("/api/videos/rendered/{filename}")
def serve_rendered_video(filename: str):
    """Annotated analyze output (track ellipses, lock highlight, ball marker)."""
    return rendered_video_response(filename)


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


@app.post("/api/pitch/calibration/preview", response_model=PitchCalibrationPreviewResponse)
def pitch_calibration_preview(
    body: PitchCalibrationSaveRequest,
) -> PitchCalibrationPreviewResponse:
    return preview_pitch_calibration(body)


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
    pitch_length_m: float | None = Form(None),
    pitch_width_m: float | None = Form(None),
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
        from backend.app.pipeline.pitch_homography import (
            MAX_BOUNDARY_POINTS,
            MIN_BOUNDARY_POINTS,
        )

        n = len(corners)
        if MIN_BOUNDARY_POINTS <= n <= MAX_BOUNDARY_POINTS:
            kwargs["image_boundary_points"] = corners
        elif n == 4:
            kwargs["image_corners"] = corners
        else:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Calibration points must be {MIN_BOUNDARY_POINTS}–"
                    f"{MAX_BOUNDARY_POINTS} (boundary) or 4 (legacy corners)."
                ),
            )
        if pitch_length_m is not None:
            kwargs["pitch_length_m"] = pitch_length_m
        if pitch_width_m is not None:
            kwargs["pitch_width_m"] = pitch_width_m
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
async def analyze_default_video(
    details: str = Form(...),
    render_video: str = Form("false"),
) -> AnalyzeResponse:
    """Analyze the server-hosted default demo video (no upload)."""
    try:
        player_details = PlayerDetails.model_validate_json(details)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid details JSON: {exc}") from exc
    video_path = str(get_default_video_file())
    render = _parse_render_video_flag(render_video)
    rendered_path: Path | None = None
    video_url: str | None = None
    if render:
        rendered_path, video_url = _allocate_rendered_output()
    logger.info(
        "Analyze started (default video): jersey=%s calibration=testmatch2 "
        "render_video=%s max_frames=%s",
        player_details.jerseyNumber,
        render,
        os.environ.get("MAX_FRAMES", "(default 6000)"),
    )
    try:
        result = await asyncio.to_thread(
            run_pipeline,
            video_path,
            player_details,
            calibration_key="testmatch2",
            output_video_path=str(rendered_path) if rendered_path else None,
        )
        logger.info(
            "Analyze finished: frames=%s target_method=%s",
            result.video.frames,
            result.target.method,
        )
        return _finalize_analyze_response(
            result,
            render_requested=render,
            rendered_path=rendered_path,
            video_url=video_url,
        )
    except HTTPException:
        _unlink_rendered(rendered_path)
        raise
    except ValueError as exc:
        _unlink_rendered(rendered_path)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _unlink_rendered(rendered_path)
        logger.exception("Analyze failed (default video)")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Check server logs for details.",
        ) from exc


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    video: UploadFile = File(...),
    details: str = Form(...),
    calibration_name: str | None = Form(None),
    render_video: str = Form("false"),
) -> AnalyzeResponse:
    logger.info(
        "Analyze request received (upload=%s); streaming body…",
        video.filename or "upload.mp4",
    )
    try:
        player_details = PlayerDetails.model_validate_json(details)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid details JSON: {exc}") from exc
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    tmp_path: str | None = None
    rendered_path: Path | None = None
    video_url: str | None = None
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

        cal_key: str | None = None
        if calibration_name and calibration_name.strip():
            cal_key = normalize_calibration_name(calibration_name.strip())
        render = _parse_render_video_flag(render_video)
        if render:
            rendered_path, video_url = _allocate_rendered_output()
        logger.info(
            "Analyze started: upload=%s bytes jersey=%s calibration=%s "
            "render_video=%s max_frames=%s",
            written,
            player_details.jerseyNumber,
            cal_key or "(default)",
            render,
            os.environ.get("MAX_FRAMES", "(default 6000)"),
        )
        result = await asyncio.to_thread(
            run_pipeline,
            tmp_path,
            player_details,
            calibration_key=cal_key,
            upload_filename=video.filename,
            output_video_path=str(rendered_path) if rendered_path else None,
        )
        logger.info(
            "Analyze finished: frames=%s target_method=%s",
            result.video.frames,
            result.target.method,
        )
        return _finalize_analyze_response(
            result,
            render_requested=render,
            rendered_path=rendered_path,
            video_url=video_url,
        )
    except HTTPException:
        _unlink_rendered(rendered_path)
        raise
    except ValueError as exc:
        _unlink_rendered(rendered_path)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _unlink_rendered(rendered_path)
        logger.exception("Analyze failed")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Check server logs for details.",
        ) from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
