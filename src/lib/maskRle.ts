import type { MaskRle, Row } from "@/types/analysis";

/** Decode row-major RLE to dense mask (index = y * w + x). */
export function decodeMaskRle(rle: MaskRle): boolean[] {
  const h = rle.size[0];
  const w = rle.size[1];
  const total = h * w;
  const flat = new Uint8Array(total);
  let idx = 0;
  let val = 0;
  for (const run of rle.counts) {
    const end = Math.min(idx + run, total);
    flat.fill(val, idx, end);
    idx = end;
    val = val === 0 ? 1 : 0;
  }
  const out = new Array<boolean>(total);
  for (let i = 0; i < total; i++) {
    out[i] = flat[i] === 1;
  }
  return out;
}

export function drawMaskOnCanvas(
  ctx: CanvasRenderingContext2D,
  mask: boolean[],
  width: number,
  height: number,
  color = "rgba(0, 200, 120, 0.5)",
) {
  const [r, g, b, a] = parseRgba(color);
  const imageData = ctx.createImageData(width, height);
  const data = imageData.data;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!mask[y * width + x]) continue;
      const i = (y * width + x) * 4;
      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
      data[i + 3] = a;
    }
  }
  ctx.putImageData(imageData, 0, 0);
}

let scratchCanvas: OffscreenCanvas | HTMLCanvasElement | null = null;
let scratchW = 0;
let scratchH = 0;

function getScratchCanvas(w: number, h: number): OffscreenCanvas | HTMLCanvasElement {
  if (scratchCanvas && scratchW === w && scratchH === h) {
    return scratchCanvas;
  }
  scratchW = w;
  scratchH = h;
  if (typeof OffscreenCanvas !== "undefined") {
    scratchCanvas = new OffscreenCanvas(w, h);
  } else {
    const el = document.createElement("canvas");
    el.width = w;
    el.height = h;
    scratchCanvas = el;
  }
  return scratchCanvas;
}

/** Draw mask at native resolution, scaled to the video canvas when sizes differ. */
export function drawMaskOnCanvasScaled(
  ctx: CanvasRenderingContext2D,
  mask: boolean[],
  maskW: number,
  maskH: number,
  destW: number,
  destH: number,
) {
  if (maskW === destW && maskH === destH) {
    drawMaskOnCanvas(ctx, mask, destW, destH);
    return;
  }
  const off = getScratchCanvas(maskW, maskH);
  const octx = off.getContext("2d");
  if (!octx) return;
  drawMaskOnCanvas(octx, mask, maskW, maskH);
  ctx.clearRect(0, 0, destW, destH);
  ctx.drawImage(off as CanvasImageSource, 0, 0, destW, destH);
}

const PLAYBACK_TIME_TOLERANCE_S = 0.05;

/** Exact frame match on pipeline index, else nearest row by timestamp `t`. */
export function rowAtPlaybackTime(
  currentTime: number,
  videoFps: number,
  maskRows: Row[],
): Row | undefined {
  if (!maskRows.length) return undefined;
  const frame = Math.round(currentTime * videoFps);
  const byFrame = maskRows.find((r) => r.frame === frame);
  if (byFrame?.mask_rle) return byFrame;

  let best: Row | undefined;
  let bestDt = PLAYBACK_TIME_TOLERANCE_S + 1;
  for (const row of maskRows) {
    if (!row.mask_rle) continue;
    const dt = Math.abs(row.t - currentTime);
    if (dt < bestDt) {
      bestDt = dt;
      best = row;
    }
  }
  return bestDt <= PLAYBACK_TIME_TOLERANCE_S ? best : undefined;
}

function parseRgba(css: string): [number, number, number, number] {
  const m = css.match(/rgba?\(([^)]+)\)/);
  if (!m) return [0, 200, 120, 128];
  const parts = m[1].split(",").map((s) => parseFloat(s.trim()));
  const alpha = parts.length >= 4 ? Math.round(parts[3] * 255) : 255;
  return [parts[0], parts[1], parts[2], alpha];
}
