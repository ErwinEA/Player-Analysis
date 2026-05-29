from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

if TYPE_CHECKING:
    from numpy.typing import NDArray


def debug_enabled() -> bool:
    return os.environ.get("DEBUG_OCR", "0").strip() in ("1", "true", "yes")


def debug_dir() -> Path:
    raw = os.environ.get("DEBUG_OCR_DIR", "debug_ocr")
    return Path(raw).resolve()


class OcrDebug:
    """Save jersey crops and append EasyOCR diagnostics when DEBUG_OCR=1."""

    def __init__(self) -> None:
        self.enabled = debug_enabled()
        self.root = debug_dir()
        self.crops_dir = self.root / "crops"
        self._crop_count = 0
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            self.crops_dir.mkdir(parents=True, exist_ok=True)
            started = datetime.now(timezone.utc).isoformat()
            self._write_log(f"# OCR debug session started {started}\n")

    def save_crop(self, frame_idx: int, track_id: int, crop: NDArray) -> Path | None:
        if not self.enabled or crop.size == 0:
            return None
        self._crop_count += 1
        path = self.crops_dir / f"f{frame_idx:06d}_t{track_id}_n{self._crop_count}.jpg"
        cv2.imwrite(str(path), crop)
        return path

    def log(
        self,
        *,
        frame_idx: int,
        track_id: int,
        number_raw: list[Any],
        parsed_number: int | None,
        number_conf: float,
        name_raw: list[Any],
        parsed_name: str | None,
        name_conf: float,
        color_score: float,
        crop_path: Path | None,
    ) -> None:
        if not self.enabled:
            return
        entry = {
            "frame": frame_idx,
            "track": track_id,
            "number_raw": number_raw,
            "parsed_number": parsed_number,
            "number_conf": round(number_conf, 4),
            "name_raw": name_raw,
            "parsed_name": parsed_name,
            "name_conf": round(name_conf, 4),
            "color_score": round(color_score, 4),
            "crop": str(crop_path) if crop_path else None,
        }
        self._write_log(json.dumps(entry) + "\n")

    def _write_log(self, text: str) -> None:
        with (self.root / "ocr.log").open("a", encoding="utf-8") as f:
            f.write(text)
