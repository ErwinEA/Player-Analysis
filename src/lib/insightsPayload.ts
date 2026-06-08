import type { PlayerDetails } from "@/components/Sidebar";
import type { AnalyzeResponse } from "@/types/analysis";
import type { InsightsRequest } from "@/types/insights";

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
    return "Ball event stats unavailable: add yolov8n_ball.pt (see backend/weights/README.md).";
  }
  if (reason === "no_calibration") {
    return "Ball event stats need pitch calibration. Calibrate the pitch, then re-run Analyze.";
  }
  if (reason === "no_lock") {
    return "Ball event stats need a confident player lock.";
  }
  return null;
}

function collectWarnings(result: AnalyzeResponse): string[] {
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

export function hasInsightsInput(result: AnalyzeResponse): boolean {
  const hasMovement =
    result.movement != null && result.movement.units === "meters";
  const hasZone = result.heatmap?.zone_summary != null;
  const hasEvents =
    result.event_counts != null && result.provenance === "inferred";
  return hasMovement || hasZone || hasEvents;
}

export function buildInsightsPayload(
  result: AnalyzeResponse,
  details: PlayerDetails,
): InsightsRequest {
  const inferred = result.inferred_events ?? [];
  const weakCount = inferred.filter((e) => e.lock_confidence === "weak").length;

  return {
    player: {
      name: details.name,
      jerseyNumber: details.jerseyNumber,
      teamName: details.teamName,
    },
    target: result.target,
    movement: result.movement ?? null,
    event_counts: result.event_counts ?? null,
    inferred_events_summary:
      inferred.length > 0
        ? {
            weak_count: weakCount,
            total_count: inferred.length,
            events: inferred.slice(0, 10),
          }
        : null,
    zone_summary: result.heatmap?.zone_summary ?? null,
    heatmap_source: result.heatmap_source ?? null,
    provenance: result.provenance ?? null,
    drive_contact_m: result.drive_contact_m ?? null,
    warnings: collectWarnings(result),
  };
}
