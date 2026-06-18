import type { PlayerDetails } from "@/components/Sidebar";
import type { AnalyzeResponse } from "@/types/analysis";
import type { InsightsRequest } from "@/types/insights";
import type { Sport } from "@/types/sport";
import { badmintonMetricsWarning } from "@/lib/badmintonMetrics";

function isConfidentPlayerLock(result: AnalyzeResponse): boolean {
  const method = result.target.method;
  return (
    result.target.track_id != null &&
    method != null &&
    method !== "none" &&
    method !== "color" &&
    result.heatmap_source !== "fallback_track"
  );
}

function eventsWarning(result: AnalyzeResponse): string | null {
  const reason = result.events_unavailable_reason;
  if (!reason) return null;
  if (reason === "no_ball_weights") {
    return "Ball event stats unavailable: add yolov8n_ball.pt (see README.md).";
  }
  if (reason === "no_calibration") {
    return "Ball event stats need pitch calibration. Calibrate the pitch, then re-run Analyze.";
  }
  if (reason === "no_lock") {
    return "Ball event stats need a confident player lock.";
  }
  return null;
}

function collectFootballWarnings(result: AnalyzeResponse): string[] {
  const warnings: string[] = [];
  const eventWarn = eventsWarning(result);
  if (eventWarn) warnings.push(eventWarn);

  if (result.calibration_skipped_reason === "positions_out_of_bounds") {
    warnings.push(
      "Pitch calibration is off: player positions fall outside the field.",
    );
  } else if (result.calibration_skipped_reason === "size_mismatch") {
    warnings.push("Pitch calibration size mismatch.");
  }

  if (!isConfidentPlayerLock(result)) {
    const method = result.target.method;
    if (method === "none" || result.target.track_id == null) {
      warnings.push(
        "Could not lock onto this jersey number. Distance and heat map may not match your player.",
      );
    } else {
      warnings.push(
        "Player lock was uncertain. Treat distance and heat map as approximate.",
      );
    }
  }

  if (result.heatmap_source === "fallback_track") {
    warnings.push("Heat map uses a best-guess track, not a confident player lock.");
  }

  return warnings;
}

const BADMINTON_HEURISTIC_CAVEAT =
  "Rally and point stats are heuristic estimates — not official scoring. Rally count may include replays; there is no line-call detection.";

function collectBadmintonWarnings(result: AnalyzeResponse): string[] {
  const warnings: string[] = [];
  if (result.warnings?.length) {
    warnings.push(...result.warnings);
  }
  const metricWarn = badmintonMetricsWarning(result);
  if (metricWarn && !warnings.includes(metricWarn)) {
    warnings.push(metricWarn);
  }
  if (result.badminton_stats?.total_rallies != null) {
    warnings.push(BADMINTON_HEURISTIC_CAVEAT);
  }
  return warnings;
}

export function hasInsightsInput(
  result: AnalyzeResponse,
  sport: Sport,
): boolean {
  const hasMovement =
    result.movement != null && result.movement.units === "meters";
  const hasZone = result.heatmap?.zone_summary != null;
  if (sport === "badminton") {
    const hasRallyStats = result.badminton_stats?.total_rallies != null;
    return hasMovement || hasZone || hasRallyStats;
  }
  const hasEvents =
    result.event_counts != null && result.provenance === "inferred";
  return hasMovement || hasZone || hasEvents;
}

export function buildInsightsPayload(
  result: AnalyzeResponse,
  details: PlayerDetails,
  sport: Sport,
): InsightsRequest {
  const inferred = result.inferred_events ?? [];
  const weakCount = inferred.filter((e) => e.lock_confidence === "weak").length;

  return {
    sport,
    court_side:
      sport === "badminton" && details.courtSide
        ? details.courtSide
        : null,
    player: {
      name: details.name,
      jerseyNumber: sport === "football" ? details.jerseyNumber : 0,
      teamName: details.teamName,
    },
    target: result.target,
    movement: result.movement ?? null,
    event_counts: sport === "football" ? (result.event_counts ?? null) : null,
    inferred_events_summary:
      sport === "football" && inferred.length > 0
        ? {
            weak_count: weakCount,
            total_count: inferred.length,
            events: inferred.slice(0, 10),
          }
        : null,
    zone_summary: result.heatmap?.zone_summary ?? null,
    heatmap_source: result.heatmap_source ?? null,
    provenance: sport === "football" ? (result.provenance ?? null) : null,
    drive_contact_m:
      sport === "football" ? (result.drive_contact_m ?? null) : null,
    badminton_stats:
      sport === "badminton" ? (result.badminton_stats ?? null) : null,
    warnings:
      sport === "badminton"
        ? collectBadmintonWarnings(result)
        : collectFootballWarnings(result),
  };
}
