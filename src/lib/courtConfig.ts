import type { Sport } from "@/types/sport";

export type CourtDimensions = {
  pitch_length_m: number;
  pitch_width_m: number;
};

export const COURT_CONFIG: Record<Sport, CourtDimensions> = {
  football: { pitch_length_m: 105.0, pitch_width_m: 68.0 },
  badminton: { pitch_length_m: 13.4, pitch_width_m: 6.1 },
};

export function courtDimensionsForSport(sport: Sport): CourtDimensions {
  return COURT_CONFIG[sport];
}

/** Static top-down court diagram served from Next.js public/. */
export const BADMINTON_COURT_IMAGE_PATH = "/badminton-court-vector.jpeg";
