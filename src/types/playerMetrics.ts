export type PlayerMetrics = {
  goals: number | null;
  shots: number | null;
  passes: number | null;
  distanceCoveredKm: number | null;
};

export const EMPTY_METRICS: PlayerMetrics = {
  goals: null,
  shots: null,
  passes: null,
  distanceCoveredKm: null,
};

export type GameplayAnalysis = {
  summary: string | null;
};
