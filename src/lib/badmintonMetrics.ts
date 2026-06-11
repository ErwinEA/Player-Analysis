import type { AnalyzeResponse } from "@/types/analysis";
import type { BadmintonMetrics } from "@/types/playerMetrics";
import { EMPTY_BADMINTON_METRICS } from "@/types/playerMetrics";

export function badmintonMetricsFromResult(
  result: AnalyzeResponse,
): BadmintonMetrics {
  const stats = result.badminton_stats;
  if (stats) {
    const winRatePct =
      stats.win_rate_pct ??
      (stats.rally_wins != null &&
      stats.total_rallies != null &&
      stats.total_rallies > 0
        ? Math.round((stats.rally_wins / stats.total_rallies) * 1000) / 10
        : null);
    return {
      rallyWins: stats.rally_wins,
      winRatePct,
      courtCoverageKm: stats.court_coverage_km,
      avgRallyDurationS: stats.avg_rally_duration_s,
      winners: stats.winners,
      movementSpeedMs: stats.movement_speed_ms,
    };
  }

  const movement = result.movement;
  if (!movement || movement.units !== "meters") {
    return EMPTY_BADMINTON_METRICS;
  }

  return {
    ...EMPTY_BADMINTON_METRICS,
    courtCoverageKm:
      Math.round((movement.distance_m / 1000) * 100) / 100,
    movementSpeedMs: Math.round(movement.avg_speed_m_s * 100) / 100,
  };
}

export function badmintonMetricsWarning(
  result: AnalyzeResponse,
): string | null {
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
    return "Rally stats need shuttle detection weights: add yolov8m_shuttlecock.pt to backend/weights (see backend/weights/README.md). Movement stats use tracking data.";
  }
  if (reason === "rally_detection_pending") {
    return "No rallies detected in this clip — rally wins, win rate, duration, and winners are unavailable. Movement stats use tracking data.";
  }
  if (result.calibration_skipped_reason === "positions_out_of_bounds") {
    return "Court calibration is off: player positions fall outside the court. Recalibrate, then re-run Analyze.";
  }
  const method = result.target.method;
  if (method === "none" || result.target.track_id == null) {
    return "Could not lock onto this player. Pick court side and shirt color, then re-run Analyze.";
  }
  if (method === "color" || result.heatmap_source === "fallback_track") {
    return "Player lock used shirt color only (court side was ambiguous). Treat movement and heat map as approximate.";
  }
  return null;
}
