from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

import cv2

from bbox_utils import crop_region

if TYPE_CHECKING:
    from numpy.typing import NDArray

_MIN_CROP_HEIGHT = int(os.environ.get("OCR_MIN_CROP_HEIGHT", "64"))


def number_crop(frame: NDArray, bbox: list[float]) -> NDArray:
    return crop_region(frame, bbox, top_pct=0.18, bot_pct=0.42, pad_x_pct=0.08)


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


class JerseyNameOCR:
    """EasyOCR reader for player name on back/torso crops only."""

    def __init__(self) -> None:
        import easyocr

        use_gpu = os.environ.get("OCR_GPU", "0") == "1"
        self.reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)

    @staticmethod
    def name_crop(frame: NDArray, bbox: list[float]) -> NDArray:
        return crop_region(frame, bbox, top_pct=0.05, bot_pct=0.30, pad_x_pct=0.10)

    def _read(self, crop: NDArray) -> list[tuple[str, float]]:
        if crop.size == 0:
            return []
        try:
            results = self.reader.readtext(crop, detail=1, paragraph=False)
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

    def read_name(self, frame: NDArray, bbox: list[float]) -> tuple[str | None, float]:
        crop = preprocess_crop(self.name_crop(frame, bbox))
        if crop.size == 0:
            return None, 0.0
        raw = self._read(crop)
        return self._parse_name(raw)
