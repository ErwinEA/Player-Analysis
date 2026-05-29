const ALLOWED_MIME_TYPES = ["image/png", "image/jpeg"] as const;

export function isValidImageFile(file: File): boolean {
  if (
    ALLOWED_MIME_TYPES.includes(
      file.type as (typeof ALLOWED_MIME_TYPES)[number],
    )
  ) {
    return true;
  }
  return /\.(png|jpe?g)$/i.test(file.name);
}

export const IMAGE_ACCEPT = "image/png,image/jpeg,.png,.jpg,.jpeg";

export const INVALID_IMAGE_MESSAGE =
  "Please select a PNG or JPEG image.";
