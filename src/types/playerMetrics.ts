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

export type BadmintonMetrics = {
  rallyWins: number | null;
  winRatePct: number | null;
  courtCoverageKm: number | null;
  avgRallyDurationS: number | null;
  winners: number | null;
  movementSpeedMs: number | null;
};

export const EMPTY_BADMINTON_METRICS: BadmintonMetrics = {
  rallyWins: null,
  winRatePct: null,
  courtCoverageKm: null,
  avgRallyDurationS: null,
  winners: null,
  movementSpeedMs: null,
};

export type GameplayAnalysis = {
  summary: string | null;
};
