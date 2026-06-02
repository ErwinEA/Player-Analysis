from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.app.pipeline.mask_rle import MaskRle


class PlayerDetails(BaseModel):
    """Mirrors PlayerDetails from src/components/Sidebar.tsx."""

    name: str = ""
    jerseyNumber: int = Field(ge=1, le=99)
    primaryJerseyColor: str = ""
    secondaryJerseyColor: str = ""
    teamName: str = ""

    @field_validator("primaryJerseyColor", "secondaryJerseyColor")
    @classmethod
    def hex_color(cls, v: str) -> str:
        if not v or not v.strip():
            return ""
        if not v.startswith("#") or len(v) not in (4, 7):
            raise ValueError("Jersey color must be a hex string like #ffffff")
        return v


def jersey_colors_configured(details: PlayerDetails) -> bool:
    return bool(
        details.primaryJerseyColor.strip() or details.secondaryJerseyColor.strip()
    )


class VideoMeta(BaseModel):
    fps: float
    width: int
    height: int
    frames: int
    duration_s: float
    frame_cap: int | None = None  # max frames processed (legacy default 5000)


class TargetMatch(BaseModel):
    jersey: int
    track_id: int | None = None
    confidence: float = 0.0
    method: str | None = None  # number+name | number | color | none
    locked_at_frame: int | None = None
    segmentation_started_at_frame: int | None = None


SegmentSourceLiteral = Literal[
    "sam",
    "sam_retry",
    "bbox_fallback",
    "reid_fallback",
    "gap_fill",
]


class Row(BaseModel):
    t: float
    frame: int
    track: int
    player: int | None = None
    x: float
    y: float
    bbox: Annotated[list[float], Field(min_length=4, max_length=4)]
    ocr_conf: float | None = None
    foot_x: float | None = None
    foot_y: float | None = None
    mask_rle: MaskRle | None = None
    mask_fallback: bool | None = None
    segment_source: SegmentSourceLiteral | None = None
    segment_track_id: int | None = None


class MovementStats(BaseModel):
    distance_m: float
    max_speed_m_s: float
    avg_speed_m_s: float
    sprint_count: int
    tracked_duration_s: float
    sample_count: int
    units: str = "meters"


class HeatmapResult(BaseModel):
    grid_cols: int
    grid_rows: int
    pitch_length_m: float
    pitch_width_m: float
    sample_count: int
    image_png_base64: str


class PlayerEventCounts(BaseModel):
    Pass: int = 0
    Shot: int = 0
    Goal: int = 0
    Drive: int = 0


class InferredBallEvent(BaseModel):
    frame: int
    kind: str
    lock_confidence: Literal["strong", "weak"]
    detection_phase: str | None = None
    possession_confidence: str | None = None


class AnalyzeResponse(BaseModel):
    video: VideoMeta
    target: TargetMatch
    rows: list[Row]
    movement: MovementStats | None = None
    heatmap: HeatmapResult | None = None
    calibration_name: str | None = None
    heatmap_source: str | None = None  # locked_target | fallback_track
    calibration_skipped_reason: str | None = None  # size_mismatch | positions_out_of_bounds
    masks_available: bool | None = None
    masks_unavailable_reason: str | None = None
    mask_row_count: int | None = None
    event_counts: PlayerEventCounts | None = None
    inferred_events: list[InferredBallEvent] | None = None
    provenance: Literal["inferred"] | None = None
    ball_samples: int | None = None
    events_unavailable_reason: str | None = None


class PitchFrameResponse(BaseModel):
    name: str
    frame_index: int
    width: int
    height: int
    image_jpeg_base64: str
    corner_labels: list[str]
    boundary_labels: list[str] = Field(default_factory=list)
    image_corners: list[list[float]] | None = None
    image_boundary_points: list[list[float]] | None = None


class PitchCalibrationSaveRequest(BaseModel):
    name: str = "testmatch2"
    frame_index: int = Field(default=100, ge=0)
    image_corners: list[list[float]] | None = None
    image_boundary_points: list[list[float]] | None = None
    point_labels: list[str] | None = None
    """Frame size for preview when the calibration video is not on the server yet."""
    image_width: int | None = Field(default=None, ge=1)
    image_height: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def image_size_pair(self) -> PitchCalibrationSaveRequest:
        w, h = self.image_width, self.image_height
        if (w is None) != (h is None):
            raise ValueError("image_width and image_height must both be set or both omitted.")
        return self

    @model_validator(mode="after")
    def require_calibration_points(self) -> PitchCalibrationSaveRequest:
        from backend.app.pipeline.pitch_homography import (
            MAX_BOUNDARY_POINTS,
            MIN_BOUNDARY_POINTS,
        )

        if self.point_labels:
            raise ValueError("Labeled landmark calibration is not supported yet.")
        boundary = self.image_boundary_points
        corners = self.image_corners
        if boundary is not None:
            n = len(boundary)
            if MIN_BOUNDARY_POINTS <= n <= MAX_BOUNDARY_POINTS:
                return self
            raise ValueError(
                f"image_boundary_points must have {MIN_BOUNDARY_POINTS}–"
                f"{MAX_BOUNDARY_POINTS} points, got {n}."
            )
        if corners is not None and len(corners) in (4, 10):
            return self
        raise ValueError(
            f"Provide image_boundary_points ({MIN_BOUNDARY_POINTS}–{MAX_BOUNDARY_POINTS}) "
            "or image_corners (4 legacy, 10 outline)."
        )


PitchCalibrationPreviewRequest = PitchCalibrationSaveRequest


class PitchCalibrationPreviewResponse(BaseModel):
    confidence: float
    coverage_pct: float
    probe_count: int
    probe_total: int
    warnings: list[str] = Field(default_factory=list)
    fitted_quad: list[list[float]]
    mode: str = "outline"
    probe_details: list[dict] = Field(default_factory=list)


class PitchCalibrationSaveResponse(BaseModel):
    name: str
    frame_index: int
    image_size: list[int]
    template_url: str
    calibration_url: str
    confidence: float | None = None
    coverage_pct: float | None = None
    warnings: list[str] = Field(default_factory=list)
