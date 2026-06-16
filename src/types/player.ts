import type { CourtSide, Sport } from "@/types/sport";

export type HexColor = `#${string}`;

export type PlayerDetails = {
  sport: Sport;
  name: string;
  jerseyNumber: number;
  courtSide: CourtSide | "";
  primaryJerseyColor: HexColor | "";
  secondaryJerseyColor: HexColor | "";
  teamName: string;
};
