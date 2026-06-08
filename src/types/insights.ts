import type {
  HeatmapZoneSummary,
  InferredBallEvent,
  MovementStats,
  PlayerEventCounts,
  TargetMatch,
} from "@/types/analysis";

export type InsightsPlayerContext = {
  name: string;
  jerseyNumber: number;
  teamName: string;
};

export type InferredEventsSummary = {
  weak_count: number;
  total_count: number;
  events: InferredBallEvent[];
};

export type InsightsRequest = {
  player: InsightsPlayerContext;
  target: TargetMatch;
  movement?: MovementStats | null;
  event_counts?: PlayerEventCounts | null;
  inferred_events_summary?: InferredEventsSummary | null;
  zone_summary?: HeatmapZoneSummary | null;
  heatmap_source?: string | null;
  provenance?: "inferred" | null;
  drive_contact_m?: number | null;
  warnings?: string[];
};

export type InsightsResponse = {
  summary: string | null;
  unavailable_reason: string | null;
};
