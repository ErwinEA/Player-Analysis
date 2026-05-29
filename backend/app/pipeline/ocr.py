from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

import cv2

from backend.app.pipeline.bbox import crop_region
from backend.app.pipeline.easyocr_shared import get_easyocr_reader
from backend.app.pipeline.jersey_number import get_jersey_classifier, number_crop

if TYPE_CHECKING:
    from numpy.typing import NDArray

_MIN_CROP_HEIGHT = int(os.environ.get("OCR_MIN_CROP_HEIGHT", "64"))


class JerseyOCR:
    """Jersey number via EfficientNet-B0 (primary) + EasyOCR (fallback); name via EasyOCR."""

    def __init__(self) -> None:
        self._number_reader = get_jersey_classifier()

    @staticmethod
    def name_crop(frame: NDArray, bbox: list[float]) -> NDArray:
        """Upper-back region where the player's name usually sits."""
        return crop_region(frame, bbox, top_pct=0.05, bot_pct=0.30, pad_x_pct=0.10)

    @staticmethod
    def number_crop(frame: NDArray, bbox: list[float]) -> NDArray:
        return number_crop(frame, bbox)

    @staticmethod
    def full_kit_crop(frame: NDArray, bbox: list[float]) -> NDArray:
        """Full jersey + shorts region (used for color matching and debugging)."""
        return crop_region(frame, bbox, top_pct=0.05, bot_pct=0.95, pad_x_pct=0.08)

    @staticmethod
    def crop_for_ocr(frame: NDArray, bbox: list[float], mask_bbox: list[float] | None) -> NDArray:
        """Kit crop from mask bounds when available, else person bbox."""
        use = mask_bbox if mask_bbox and len(mask_bbox) >= 4 else bbox
        return JerseyOCR.full_kit_crop(frame, use)

    @staticmethod
    def jersey_crop(frame: NDArray, bbox: list[float]) -> NDArray:
        """Back-compat alias used by debug tooling: combined name+number band."""
        return crop_region(frame, bbox, top_pct=0.05, bot_pct=0.75, pad_x_pct=0.10)

    @staticmethod
    def preprocess_crop(crop: NDArray) -> NDArray:
        if crop.size == 0:
            return crop
        out = crop
        h = out.shape[0]
        if h < _MIN_CROP_HEIGHT:
            scale = _MIN_CROP_HEIGHT / h
            out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _read_name_text(self, crop: NDArray) -> list[tuple[str, float]]:
        if crop.size == 0:
            return []
        try:
            results = get_easyocr_reader().readtext(crop, detail=1, paragraph=False)
        except Exception as exc:
            return [(f"error:{exc}", 0.0)]
        out: list[tuple[str, float]] = []
        for item in results or []:
            if not item or len(item) < 3:
                continue
            out.append((str(item[1]), float(item[2])))
        return out

    @staticmethod
    def _parse_name(raw: list[tuple[str, float]]) -> tuple[str | None, float]:
        best_text: str | None = None
        best_conf = 0.0
        for text, conf in raw:
            letters = re.sub(r"[^A-Za-z]", "", text)
            if len(letters) >= 3 and conf > best_conf:
                best_text = letters.upper()
                best_conf = conf
        return best_text, best_conf

    def read_number(self, frame: NDArray, bbox: list[float]) -> tuple[int | None, float]:
        num, conf, _raw = self.read_number_detailed(frame, bbox)
        return num, conf

    def read_number_detailed(
        self,
        frame: NDArray,
        bbox: list[float],
        *,
        target_number: int | None = None,
    ) -> tuple[int | None, float, list[tuple[str, float]]]:
        return self._number_reader.read_number_detailed(
            frame, bbox, target_number=target_number
        )

    def read_name_detailed(
        self, frame: NDArray, bbox: list[float]
    ) -> tuple[str | None, float, list[tuple[str, float]]]:
        crop = self.preprocess_crop(self.name_crop(frame, bbox))
        if crop.size == 0:
            return None, 0.0, []
        raw = self._read_name_text(crop)
        text, conf = self._parse_name(raw)
        return text, conf, raw
