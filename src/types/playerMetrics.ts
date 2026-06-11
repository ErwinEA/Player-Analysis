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
  totalRallies: number | null;
  avgRallyDurationS: number | null;
  longestRallyS: number | null;
  pointsWonOnServe: number | null;
  pointsWonOnReturn: number | null;
  totalPointsWon: number | null;
  pointsIn: number | null;
  pointsOut: number | null;
};

export const EMPTY_BADMINTON_METRICS: BadmintonMetrics = {
  totalRallies: null,
  avgRallyDurationS: null,
  longestRallyS: null,
  pointsWonOnServe: null,
  pointsWonOnReturn: null,
  totalPointsWon: null,
  pointsIn: null,
  pointsOut: null,
};

export type GameplayAnalysis = {
  summary: string | null;
};
