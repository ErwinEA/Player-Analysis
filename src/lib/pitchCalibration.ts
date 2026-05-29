/** Client mirror of backend decagon_to_quad for calibration preview. */

export const BOUNDARY_POINT_COUNT = 10;

export const BOUNDARY_CORNER_INDICES = [0, 3, 6, 9] as const;

export const BOUNDARY_LABELS = [
  "Top left corner",
  "Top line (left)",
  "Top line (center)",
  "Top right corner",
  "Right side (upper)",
  "Right side (lower)",
  "Bottom right corner",
  "Bottom line (right)",
  "Bottom line (left)",
  "Bottom left corner",
] as const;

export type CalPoint = { x: number; y: number };

function orderCornersTlTrBrBl(pts: CalPoint[]): CalPoint[] {
  const p = pts.slice(0, 4);
  const bySum = p.map((pt) => pt.x + pt.y);
  const byDiff = p.map((pt) => pt.x - pt.y);
  const used = new Set<number>();
  const picks: number[] = [];
  for (const key of [
    bySum.indexOf(Math.min(...bySum)),
    byDiff.indexOf(Math.max(...byDiff)),
    bySum.indexOf(Math.max(...bySum)),
    byDiff.indexOf(Math.min(...byDiff)),
  ]) {
    if (!used.has(key)) {
      used.add(key);
      picks.push(key);
    }
  }
  for (let i = 0; i < 4; i++) {
    if (!used.has(i)) picks.push(i);
  }
  return picks.slice(0, 4).map((i) => p[i]);
}

export function decagonToQuad(boundary: CalPoint[]): CalPoint[] {
  if (boundary.length < 4) return [];
  if (boundary.length === BOUNDARY_POINT_COUNT) {
    const corners = BOUNDARY_CORNER_INDICES.map((i) => boundary[i]);
    return orderCornersTlTrBrBl(corners);
  }
  if (boundary.length === 4) {
    return boundary.slice(0, 4);
  }

  const center = {
    x: boundary.reduce((s, p) => s + p.x, 0) / boundary.length,
    y: boundary.reduce((s, p) => s + p.y, 0) / boundary.length,
  };
  const ordered = [...boundary].sort(
    (a, b) =>
      Math.atan2(a.y - center.y, a.x - center.x) -
      Math.atan2(b.y - center.y, b.x - center.x),
  );
  const n = ordered.length;
  const quad: CalPoint[] = [];
  for (let i = 0; i < 4; i++) {
    const start = Math.round((i * n) / 4);
    const end = Math.round(((i + 1) * n) / 4);
    const arc = ordered.slice(start, end > start ? end : start + 1);
    let best = arc[0];
    let bestD = -1;
    for (const pt of arc) {
      const d = (pt.x - center.x) ** 2 + (pt.y - center.y) ** 2;
      if (d > bestD) {
        bestD = d;
        best = pt;
      }
    }
    quad.push(best);
  }
  return orderCornersTlTrBrBl(quad);
}

/** Place legacy 4 warp corners into a 10-slot outline (corners only). */
export function legacyCornersToBoundary(corners: CalPoint[]): CalPoint[] {
  const slots: CalPoint[] = Array.from({ length: BOUNDARY_POINT_COUNT }, () => ({
    x: 0,
    y: 0,
  }));
  BOUNDARY_CORNER_INDICES.forEach((idx, i) => {
    if (corners[i]) slots[idx] = corners[i];
  });
  return slots;
}

export function isLegacyBoundaryOutline(points: CalPoint[]): boolean {
  if (points.length !== BOUNDARY_POINT_COUNT) return false;
  const filled = new Set(
    points
      .map((p, i) => (p.x !== 0 || p.y !== 0 ? i : -1))
      .filter((i) => i >= 0),
  );
  const cornerSet = new Set(BOUNDARY_CORNER_INDICES);
  if (filled.size !== cornerSet.size) return false;
  for (const i of filled) {
    if (!cornerSet.has(i as (typeof BOUNDARY_CORNER_INDICES)[number])) {
      return false;
    }
  }
  return true;
}
