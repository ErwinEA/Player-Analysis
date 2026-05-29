const ALLOWED_MIME_TYPES = ["video/mp4", "video/quicktime"] as const;

export function isValidVideoFile(file: File): boolean {
  if (ALLOWED_MIME_TYPES.includes(file.type as (typeof ALLOWED_MIME_TYPES)[number])) {
    return true;
  }
  return /\.(mp4|mov)$/i.test(file.name);
}

export const VIDEO_ACCEPT =
  "video/mp4,video/quicktime,.mp4,.mov";

export const INVALID_VIDEO_MESSAGE =
  "Please select an MP4 or MOV video file.";
