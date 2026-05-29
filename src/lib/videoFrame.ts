import type { PitchFrameResponse } from "@/lib/api";

/** Assumed FPS when mapping seek time ↔ frame index (matches backend cv2 seek). */
export const CALIBRATION_ASSUMED_FPS = 25;

export function calibrationKeyFromFilename(filename: string): string {
  const stem = filename.replace(/\.[^.]+$/, "");
  const safe = stem
    .replace(/[^A-Za-z0-9_-]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_|_$/g, "");
  return safe || "upload";
}

/** Seek ~15% into clip (min 2s), capped — often a wide tactical view of the pitch. */
export function suggestedCalibrationSeekSeconds(durationSec: number): number {
  if (!Number.isFinite(durationSec) || durationSec <= 0) return 4;
  const target = Math.max(2, durationSec * 0.15);
  return Math.min(target, Math.max(0, durationSec - 0.5));
}

export function seekSecondsToFrameIndex(seconds: number): number {
  return Math.max(0, Math.round(seconds * CALIBRATION_ASSUMED_FPS));
}

export function frameIndexToSeekSeconds(frameIndex: number): number {
  return frameIndex / CALIBRATION_ASSUMED_FPS;
}

export async function getVideoDuration(file: File): Promise<number> {
  const url = URL.createObjectURL(file);
  try {
    const video = document.createElement("video");
    video.preload = "metadata";
    video.src = url;
    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error("Could not read video metadata"));
    });
    return video.duration;
  } finally {
    URL.revokeObjectURL(url);
  }
}

export async function extractVideoFrameAtTime(
  file: File,
  seekSeconds: number,
  calibrationName: string,
): Promise<PitchFrameResponse> {
  const url = URL.createObjectURL(file);
  try {
    const video = document.createElement("video");
    video.preload = "auto";
    video.muted = true;
    video.playsInline = true;
    video.src = url;
    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error("Could not load video"));
    });
    const clamped = Math.min(
      Math.max(0, seekSeconds),
      Math.max(0, video.duration - 0.05),
    );
    video.currentTime = clamped;
    await new Promise<void>((resolve, reject) => {
      video.onseeked = () => resolve();
      video.onerror = () => reject(new Error("Could not seek video"));
    });
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) {
      throw new Error("Video has no frame dimensions");
    }
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas not supported");
    ctx.drawImage(video, 0, 0, width, height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
    const base64 = dataUrl.split(",")[1] ?? "";
    const frameIndex = seekSecondsToFrameIndex(clamped);
    return {
      name: calibrationName,
      frame_index: frameIndex,
      width,
      height,
      image_jpeg_base64: base64,
      corner_labels: [
        "top_left",
        "top_right",
        "bottom_right",
        "bottom_left",
      ],
      image_corners: null,
    };
  } finally {
    URL.revokeObjectURL(url);
  }
}
