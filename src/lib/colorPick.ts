export type HexColor = `#${string}`;

export function rgbToHex(r: number, g: number, b: number): HexColor {
  const clamp = (n: number) =>
    Math.max(0, Math.min(255, Math.round(n)))
      .toString(16)
      .padStart(2, "0");
  return `#${clamp(r)}${clamp(g)}${clamp(b)}`;
}

/** Average RGB in a square neighborhood (helps under compression / lighting). */
export function sampleColorFromCanvas(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  canvasWidth: number,
  canvasHeight: number,
  radius = 2,
): HexColor {
  const left = Math.max(0, Math.floor(x) - radius);
  const top = Math.max(0, Math.floor(y) - radius);
  const right = Math.min(canvasWidth, Math.floor(x) + radius + 1);
  const bottom = Math.min(canvasHeight, Math.floor(y) + radius + 1);
  const width = Math.max(1, right - left);
  const height = Math.max(1, bottom - top);
  const { data } = ctx.getImageData(left, top, width, height);

  let r = 0;
  let g = 0;
  let b = 0;
  let count = 0;
  for (let i = 0; i < data.length; i += 4) {
    const alpha = data[i + 3];
    if (alpha === 0) continue;
    r += data[i];
    g += data[i + 1];
    b += data[i + 2];
    count += 1;
  }
  if (count === 0) return "#000000";
  return rgbToHex(r / count, g / count, b / count);
}
