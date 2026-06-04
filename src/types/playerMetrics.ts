export type PlayerMetrics = {
  goals: number | null;
  shots: number | null;
  passes: number | null;
  drives: number | null;
  distanceCoveredKm: number | null;
  driveContactM: number | null;
};

export const EMPTY_METRICS: PlayerMetrics = {
  goals: null,
  shots: null,
  passes: null,
  drives: null,
  distanceCoveredKm: null,
  driveContactM: null,
};

export type GameplayAnalysis = {
  summary: string | null;
};
