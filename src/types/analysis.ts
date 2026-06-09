export type VideoMeta = {
  fps: number;
  width: number;
  height: number;
  frames: number;
  duration_s: number;
  frame_cap?: number | null;
  /** Leading frames skipped by pipeline (FRAME_START); offsets mask sync on rendered clip. */
  frame_start?: number;
};

export type MaskRle = {
  size: [number, number];
  counts: number[];
};

export type SegmentSource =
  | "sam"
  | "sam_retry"
  | "bbox_fallback"
  | "reid_fallback"
  | "gap_fill";

export type TargetMatch = {
  jersey: number;
  track_id: number | null;
  confidence: number;
  method: string | null;
  locked_at_frame: number | null;
  segmentation_started_at_frame?: number | null;
};

export type Row = {
  t: number;
  frame: number;
  track: number;
  player: number | null;
  x: number;
  y: number;
  bbox: [number, number, number, number];
  ocr_conf: number | null;
  foot_x?: number | null;
  foot_y?: number | null;
  mask_rle?: MaskRle | null;
  mask_fallback?: boolean | null;
  segment_source?: SegmentSource | null;
  segment_track_id?: number | null;
};

export type MovementStats = {
  distance_m: number;
  max_speed_m_s: number;
  avg_speed_m_s: number;
  sprint_count: number;
  tracked_duration_s: number;
  sample_count: number;
  units: string;
};

export type HeatmapZoneSummary = {
  defensive_third_pct: number;
  middle_third_pct: number;
  attacking_third_pct: number;
  left_pct: number;
  center_pct: number;
  right_pct: number;
  central_pct: number;
  wide_pct: number;
  hottest_x_m: number | null;
  hottest_y_m: number | null;
};

export type HeatmapResult = {
  grid_cols: number;
  grid_rows: number;
  pitch_length_m: number;
  pitch_width_m: number;
  sample_count: number;
  image_png_base64: string;
  zone_summary?: HeatmapZoneSummary | null;
};

export type PlayerEventCounts = {
  Pass: number;
  Shot: number;
  Goal: number;
  Drive: number;
};

export type InferredBallEvent = {
  frame: number;
  kind: string;
  lock_confidence: "strong" | "weak";
  detection_phase?: string | null;
  possession_confidence?: string | null;
};

export type AnalyzeResponse = {
  video: VideoMeta;
  target: TargetMatch;
  rows: Row[];
  movement?: MovementStats | null;
  heatmap?: HeatmapResult | null;
  calibration_name?: string | null;
  heatmap_source?: "locked_target" | "fallback_track" | null;
  calibration_skipped_reason?: string | null;
  masks_available?: boolean | null;
  masks_unavailable_reason?: string | null;
  mask_row_count?: number | null;
  event_counts?: PlayerEventCounts | null;
  inferred_events?: InferredBallEvent[] | null;
  provenance?: "inferred" | null;
  ball_samples?: number | null;
  events_unavailable_reason?: string | null;
  drive_contact_m?: number | null;
  video_url?: string | null;
  video_unavailable_reason?: string | null;
};
