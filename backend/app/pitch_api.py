from __future__ import annotations

import base64
import re
from pathlib import Path

import cv2
from fastapi import HTTPException
from fastapi.responses import FileResponse

from backend.app.pipeline.pitch_homography import (
    BOUNDARY_LABELS,
    BOUNDARY_POINT_COUNT,
    CalibrationDiagnostics,
    CORNER_LABELS,
    build_calibration_with_diagnostics,
    default_calibration_dir,
    load_calibration,
    render_pitch_template,
    save_calibration,
)
from backend.app.schemas import (
    PitchCalibrationPreviewResponse,
    PitchCalibrationSaveRequest,
    PitchCalibrationSaveResponse,
    PitchFrameResponse,
)
from backend.app.video_store import (
    DEFAULT_FRAME_INDEX,
    DEFAULT_VIDEO_NAME,
    get_default_video_file,
)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def calibration_key_from_filename(filename: str) -> str:
    """Safe calibration name for saving pitch JSON / legacy upload (not auto-used on analyze)."""
    stem = Path(filename).stem
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", stem)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "upload"


def default_pitch_calibration_name() -> str:
    import os

    return os.environ.get("PITCH_CALIBRATION_NAME", "testmatch2").strip() or "testmatch2"


def calibration_keys_for_run(*, explicit_key: str | None = None) -> tuple[str, ...]:
    """Calibration lookup order shared by CLI and HTTP analyze.

    Matches ``python -m backend.cli``: use ``PITCH_CALIBRATION_NAME`` (default
    ``testmatch2``). Only when the client passes an explicit key (e.g. legacy UI
    after saving pitch corners for that video) try that name first, then fall back
  to the default.

    Does **not** auto-select ``{video_stem}.json`` from the upload filename — that
    caused analyze to diverge from CLI when a stale per-stem calibration existed.
    """
    default = default_pitch_calibration_name()
    if explicit_key:
        safe = _safe_name_or_none(explicit_key)
        if safe and safe != default:
            return (safe, default)
    return (default,)


_LEGACY_VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".webm")


def _stored_calibration_video_path(name: str) -> Path | None:
    out_dir = default_calibration_dir()
    for suffix in _LEGACY_VIDEO_SUFFIXES:
        path = out_dir / f"{name}{suffix}"
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def legacy_calibration_video_stored(name: str) -> bool:
    return _stored_calibration_video_path(_validate_name(name)) is not None


def legacy_calibration_video_dest(name: str, suffix: str) -> Path:
    safe = _validate_name(name)
    ext = suffix if suffix.startswith(".") else f".{suffix}"
    return default_calibration_dir() / f"{safe}{ext}"


def normalize_calibration_name(name: str) -> str:
    return _validate_name(name)


def _validate_name(name: str) -> str:
    if not _SAFE_NAME.fullmatch(name):
        raise HTTPException(
            status_code=400,
            detail="Invalid calibration name. Use letters, digits, underscore, or hyphen only.",
        )
    return name


def _resolve_under_calibration_dir(path: Path) -> Path:
    base = default_calibration_dir().resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(base):
        raise HTTPException(status_code=400, detail="Invalid calibration path.")
    return resolved


def _template_path(name: str) -> Path:
    return _resolve_under_calibration_dir(default_calibration_dir() / f"{name}_topview_template.png")


def _calibration_path(name: str) -> Path:
    return _resolve_under_calibration_dir(default_calibration_dir() / f"{name}.json")


def load_calibration_by_name(name: str):
    """Load pitch calibration by safe name; raises HTTPException if missing."""
    cal_path = _calibration_path(_validate_name(name))
    if not cal_path.is_file():
        raise HTTPException(status_code=404, detail=f"Pitch calibration not found for '{name}'")
    return load_calibration(cal_path)


def _safe_name_or_none(name: str) -> str | None:
    if not _SAFE_NAME.fullmatch(name):
        return None
    return name


def resolve_calibration(*names: str | None):
    """Return (calibration, matched_name) for the first existing safe name."""
    seen: set[str] = set()
    for raw in names:
        if not raw:
            continue
        safe = _safe_name_or_none(raw)
        if safe is None or safe in seen:
            continue
        seen.add(safe)
        cal_path = _calibration_path(safe)
        if cal_path.is_file():
            return load_calibration(cal_path), safe
    return None, None


def get_pitch_template_file(name: str = "testmatch2") -> Path:
    name = _validate_name(name)
    path = _template_path(name)
    if path.is_file():
        return path
    # Generate on demand if calibration exists but template PNG missing
    cal_path = _calibration_path(name)
    if cal_path.is_file():
        img = render_pitch_template(
            length_m=load_calibration(cal_path).pitch_length_m,
            width_m=load_calibration(cal_path).pitch_width_m,
        )
        import cv2

        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)
        return path
    raise HTTPException(status_code=404, detail=f"Pitch template not found for '{name}'")


def pitch_template_response(name: str = "testmatch2") -> FileResponse:
    path = get_pitch_template_file(name)
    return FileResponse(path, media_type="image/png", filename=f"{name}_pitch.png")


def get_pitch_calibration(name: str = "testmatch2") -> dict:
    name = _validate_name(name)
    cal = load_calibration_by_name(name)
    return {
        "name": cal.name,
        "pitch_length_m": cal.pitch_length_m,
        "pitch_width_m": cal.pitch_width_m,
        "frame_index": cal.frame_index,
        "image_size": list(cal.image_size),
        "template_url": f"/api/pitch/template?name={name}",
    }


def _load_video_frame(path: Path, frame_index: int) -> tuple[cv2.typing.MatLike, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise HTTPException(status_code=404, detail=f"Could not open video: {path}")
    try:
        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read frame {frame_index}",
            )
        h, w = frame.shape[:2]
        return frame, w, h
    finally:
        cap.release()


def _encode_jpeg_b64(frame) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode frame JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def get_pitch_frame(
    *,
    name: str = DEFAULT_VIDEO_NAME,
    frame_index: int = DEFAULT_FRAME_INDEX,
) -> PitchFrameResponse:
    name = _validate_name(name)
    if stored := _stored_calibration_video_path(name):
        video_path = stored
    else:
        video_path = get_default_video_file()

    frame, width, height = _load_video_frame(video_path, frame_index)
    existing_corners: list[list[float]] | None = None
    existing_boundary: list[list[float]] | None = None
    cal_path = _calibration_path(name)
    if cal_path.is_file():
        cal = load_calibration(cal_path)
        if cal.frame_index == frame_index and cal.image_size == (width, height):
            existing_corners = cal.image_corners
            existing_boundary = cal.image_boundary_points

    return PitchFrameResponse(
        name=name,
        frame_index=frame_index,
        width=width,
        height=height,
        image_jpeg_base64=_encode_jpeg_b64(frame),
        corner_labels=list(CORNER_LABELS),
        boundary_labels=list(BOUNDARY_LABELS),
        image_corners=existing_corners,
        image_boundary_points=existing_boundary,
    )


def _diagnostics_to_preview_response(
    diag: CalibrationDiagnostics,
) -> PitchCalibrationPreviewResponse:
    return PitchCalibrationPreviewResponse(
        confidence=diag.confidence,
        coverage_pct=round(diag.coverage_pct * 100.0, 1),
        probe_count=diag.probe_count,
        probe_total=diag.probe_total,
        warnings=list(diag.warnings),
        fitted_quad=[list(c) for c in diag.fitted_quad],
        mode=diag.mode,
        probe_details=list(diag.probe_details),
    )


def _calibration_from_request(
    name: str,
    body: PitchCalibrationSaveRequest,
    *,
    video_path: Path,
) -> tuple[object, CalibrationDiagnostics, object, int, int]:
    frame, width, height = _load_video_frame(video_path, body.frame_index)
    try:
        if body.image_boundary_points is not None:
            cal, diag = build_calibration_with_diagnostics(
                name,
                image_boundary_points=body.image_boundary_points,
                video_path=str(video_path),
                frame_index=body.frame_index,
                image_size=(width, height),
            )
        else:
            points = body.image_corners or []
            if len(points) == BOUNDARY_POINT_COUNT:
                cal, diag = build_calibration_with_diagnostics(
                    name,
                    image_boundary_points=points,
                    video_path=str(video_path),
                    frame_index=body.frame_index,
                    image_size=(width, height),
                )
            else:
                cal, diag = build_calibration_with_diagnostics(
                    name,
                    image_corners=points,
                    video_path=str(video_path),
                    frame_index=body.frame_index,
                    image_size=(width, height),
                )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return cal, diag, frame, width, height


def _preview_from_client_image_size(
    name: str,
    body: PitchCalibrationSaveRequest,
) -> PitchCalibrationPreviewResponse:
    assert body.image_width is not None and body.image_height is not None
    try:
        if body.image_boundary_points is not None:
            _cal, diag = build_calibration_with_diagnostics(
                name,
                image_boundary_points=body.image_boundary_points,
                video_path="",
                frame_index=body.frame_index,
                image_size=(body.image_width, body.image_height),
            )
        else:
            points = body.image_corners or []
            if len(points) == BOUNDARY_POINT_COUNT:
                _cal, diag = build_calibration_with_diagnostics(
                    name,
                    image_boundary_points=points,
                    video_path="",
                    frame_index=body.frame_index,
                    image_size=(body.image_width, body.image_height),
                )
            else:
                _cal, diag = build_calibration_with_diagnostics(
                    name,
                    image_corners=points,
                    video_path="",
                    frame_index=body.frame_index,
                    image_size=(body.image_width, body.image_height),
                )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _diagnostics_to_preview_response(diag)


def preview_pitch_calibration(
    body: PitchCalibrationSaveRequest,
) -> PitchCalibrationPreviewResponse:
    name = _validate_name(body.name)
    if body.image_width is not None and body.image_height is not None:
        return _preview_from_client_image_size(name, body)
    if stored := _stored_calibration_video_path(name):
        video_path = stored
    else:
        video_path = get_default_video_file()
    _cal, diag, _frame, _w, _h = _calibration_from_request(
        name, body, video_path=video_path
    )
    return _diagnostics_to_preview_response(diag)


def save_pitch_calibration(body: PitchCalibrationSaveRequest) -> PitchCalibrationSaveResponse:
    name = _validate_name(body.name)

    if stored := _stored_calibration_video_path(name):
        video_path = stored
    else:
        video_path = get_default_video_file()
    cal, diag, frame, width, height = _calibration_from_request(
        name, body, video_path=video_path
    )

    out_dir = default_calibration_dir()
    save_calibration(cal, out_dir / f"{name}.json")

    template = render_pitch_template(
        length_m=cal.pitch_length_m,
        width_m=cal.pitch_width_m,
    )
    template_path = out_dir / f"{name}_topview_template.png"
    cv2.imwrite(str(template_path), template)

    from backend.app.pipeline.pitch_homography import draw_corners_overlay

    overlay_path = out_dir / f"{name}_corners.jpg"
    cv2.imwrite(
        str(overlay_path),
        draw_corners_overlay(
            frame,
            cal.image_corners,
            boundary_points=cal.image_boundary_points,
        ),
    )

    return PitchCalibrationSaveResponse(
        name=name,
        frame_index=body.frame_index,
        image_size=[width, height],
        template_url=f"/api/pitch/template?name={name}",
        calibration_url=f"/api/pitch/calibration?name={name}",
        confidence=diag.confidence,
        coverage_pct=round(diag.coverage_pct * 100.0, 1),
        warnings=list(diag.warnings),
    )


def save_pitch_calibration_upload(
    *,
    name: str,
    frame_index: int,
    image_corners: list[list[float]] | None = None,
    image_boundary_points: list[list[float]] | None = None,
    video_path: Path,
) -> PitchCalibrationSaveResponse:
    """Persist legacy corner calibration using an on-disk video (already uploaded)."""
    name = _validate_name(name)

    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise HTTPException(status_code=404, detail="Legacy video not found on server")

    body = PitchCalibrationSaveRequest(
        name=name,
        frame_index=frame_index,
        image_corners=image_corners,
        image_boundary_points=image_boundary_points,
    )
    out_dir = default_calibration_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    cal, diag, frame, width, height = _calibration_from_request(
        name, body, video_path=video_path
    )

    save_calibration(cal, out_dir / f"{name}.json")

    template = render_pitch_template(
        length_m=cal.pitch_length_m,
        width_m=cal.pitch_width_m,
    )
    template_path = out_dir / f"{name}_topview_template.png"
    cv2.imwrite(str(template_path), template)

    from backend.app.pipeline.pitch_homography import draw_corners_overlay

    overlay_path = out_dir / f"{name}_corners.jpg"
    cv2.imwrite(
        str(overlay_path),
        draw_corners_overlay(
            frame,
            cal.image_corners,
            boundary_points=cal.image_boundary_points,
        ),
    )

    return PitchCalibrationSaveResponse(
        name=name,
        frame_index=frame_index,
        image_size=[width, height],
        template_url=f"/api/pitch/template?name={name}",
        calibration_url=f"/api/pitch/calibration?name={name}",
        confidence=diag.confidence,
        coverage_pct=round(diag.coverage_pct * 100.0, 1),
        warnings=list(diag.warnings),
    )
