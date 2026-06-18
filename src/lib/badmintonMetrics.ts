import type { AnalyzeResponse } from "@/types/analysis";
import type { BadmintonMetrics } from "@/types/playerMetrics";
import { EMPTY_BADMINTON_METRICS } from "@/types/playerMetrics";

export function badmintonMetricsFromResult(
  result: AnalyzeResponse,
): BadmintonMetrics {
  const stats = result.badminton_stats;
  if (!stats) {
    return EMPTY_BADMINTON_METRICS;
  }
  return {
    totalRallies: stats.total_rallies,
    avgRallyDurationS: stats.avg_rally_duration_s,
    longestRallyS: stats.longest_rally_duration_s,
    pointsWonOnServe: stats.points_won_on_serve,
    pointsWonOnReturn: stats.points_won_on_return,
    totalPointsWon: stats.total_points_won,
    pointsIn: stats.points_in,
    pointsOut: stats.points_out,
  };
}

export function badmintonMetricsWarning(
  result: AnalyzeResponse,
): string | null {
  if (result.warnings?.length) {
    return result.warnings[0];
  }
  if (result.calibration_skipped_reason === "court_dimension_mismatch") {
    return "Saved court calibration uses football dimensions (105 m × 68 m). Click Calibrate court layout, save corners for this video in Badminton mode (13.4 m × 6.1 m), then re-run Analyze.";
  }
  const reason = result.badminton_stats_unavailable_reason;
  if (reason === "no_calibration") {
    return "Rally stats need court calibration. Calibrate the court, then re-run Analyze.";
  }
  if (reason === "no_lock") {
    return "Rally stats need a confident player lock via court side or shirt color.";
  }
  if (reason === "no_shuttle_weights") {
    return "Rally stats need shuttle detection weights: add yolov8m_shuttlecock.pt to backend/weights (see README.md).";
  }
  if (reason === "rallies_replay_filtered") {
    return "Rallies were detected but excluded as likely replay/highlight footage — try a continuous match clip.";
  }
  if (reason === "rally_detection_pending") {
    return "No rallies detected — check shuttle visibility, calibration, and clip type (highlights often fail).";
  }
  if (result.target.method === "court_side_single") {
    return "Player lock used court side with only one visible player (lower confidence). Treat heat map as approximate.";
  }
  if (result.calibration_skipped_reason === "positions_out_of_bounds") {
    return "Court calibration is off: player positions fall outside the court. Recalibrate, then re-run Analyze.";
  }
  const method = result.target.method;
  if (method === "none" || result.target.track_id == null) {
    return "Could not lock onto this player. Pick court side and shirt color, then re-run Analyze.";
  }
  if (method === "color" || result.heatmap_source === "fallback_track") {
    return "Player lock used shirt color only (court side was ambiguous). Treat heat map as approximate.";
  }
  return null;
}
