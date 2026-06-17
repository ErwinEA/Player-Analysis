"""EfficientNet-B0 jersey number classifier with EasyOCR digits fallback."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

from backend.app.pipeline.bbox import crop_region
from backend.app.pipeline.easyocr_shared import get_easyocr_reader
from backend.app.pipeline.jersey_weights import resolve_jersey_meta, resolve_jersey_weights

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_MIN_CONF = float(os.environ.get("JERSEY_CLS_MIN_CONF", "0.75"))
_TARGET_MIN_CONF = float(os.environ.get("JERSEY_TARGET_MIN_CONF", "0.55"))
_SOFT_MIN_CONF = float(os.environ.get("JERSEY_SOFT_MIN_CONF", "0.25"))
_NUMBER_MIN_CONF = float(os.environ.get("NUMBER_MIN_CONF", "0.40"))
_CLASSIFIER_MIN_CONF = float(os.environ.get("CLASSIFIER_MIN_CONF", "0.5"))
_OCR_MIN_CROP_HEIGHT = int(os.environ.get("OCR_MIN_CROP_HEIGHT", "64"))
_DEFAULT_INPUT_SIZE = 224
_DEFAULT_CLASS_NAMES = ["unknown"] + [str(n) for n in range(1, 100)]


def number_crop(frame: NDArray, bbox: list[float]) -> NDArray:
    """Tighter back band for digits (aligned with detection_test)."""
    return crop_region(frame, bbox, top_pct=0.18, bot_pct=0.42, pad_x_pct=0.08)


def preprocess_crop_for_ocr(crop: NDArray) -> NDArray:
    if crop.size == 0:
        return crop
    out = crop
    h = out.shape[0]
    if h < _OCR_MIN_CROP_HEIGHT:
        scale = _OCR_MIN_CROP_HEIGHT / h
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def resolve_jersey_device_str() -> str:
    """Resolved jersey classifier device string (``mps``, ``cuda``, or ``cpu``)."""
    pref = os.environ.get("JERSEY_CLS_DEVICE", "auto").lower()
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if pref == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_inference_device() -> torch.device:
    resolved = resolve_jersey_device_str()
    pref = os.environ.get("JERSEY_CLS_DEVICE", "auto").lower()
    if pref == "mps" and resolved == "cpu":
        logger.error(
            "JERSEY_CLS_DEVICE=mps but MPS is unavailable (macOS 14+, PyTorch 2.0+)"
        )
    elif pref == "cuda" and resolved == "cpu":
        logger.error("JERSEY_CLS_DEVICE=cuda but CUDA is unavailable")
    return torch.device(resolved)


def _parse_digits_from_ocr(
    results: list,
) -> tuple[int | None, float, list[tuple[str, float]]]:
    raw: list[tuple[str, float]] = []
    best_num: int | None = None
    best_conf = 0.0
    for item in results or []:
        if not item or len(item) < 3:
            continue
        text, conf = str(item[1]), float(item[2])
        raw.append((text, conf))
        digits = re.sub(r"\D", "", text)
        if 1 <= len(digits) <= 2:
            try:
                num = int(digits)
            except ValueError:
                continue
            if not (1 <= num <= 99):
                continue
            if conf > best_conf:
                best_num = num
                best_conf = conf
    return best_num, best_conf, raw


class _DigitOCRFallback:
    """EasyOCR digits-only when classifier is unavailable or low confidence."""

    def read(self, crop: NDArray) -> tuple[int | None, float, list[tuple[str, float]]]:
        if crop.size == 0:
            return None, 0.0, []
        try:
            results = get_easyocr_reader().readtext(
                crop, detail=1, paragraph=False, allowlist="0123456789"
            )
        except Exception as exc:
            return None, 0.0, [(f"error:{exc}", 0.0)]
        return _parse_digits_from_ocr(results)


def _build_backbone(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


class JerseyNumberClassifier:
    """EfficientNet-B0 jersey classifier (1–99 + unknown) with EasyOCR fallback."""

    def __init__(self, weights_path: str | None = None) -> None:
        self.weights_path = resolve_jersey_weights(weights_path)
        self.class_names: list[str] = list(_DEFAULT_CLASS_NAMES)
        self.model_name = "efficientnet_b0"
        self.input_size = _DEFAULT_INPUT_SIZE
        self.model: nn.Module | None = None
        self.device = _resolve_inference_device()
        self._ocr_fallback = _DigitOCRFallback()
        self._transform = self._make_transform(self.input_size)

        if self.weights_path is not None:
            self._load_weights(self.weights_path)
        else:
            logger.warning(
                "Jersey classifier weights not found (set JERSEY_WEIGHTS or place "
                "jersey_number_b0.pt under backend/weights/) — EasyOCR fallback only."
            )

    @staticmethod
    def _make_transform(input_size: int) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _load_weights(self, path: Path) -> None:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            try:
                logger.warning(
                    "weights_only load failed for %s; retrying full checkpoint (trusted local path)",
                    path,
                )
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as exc:
                logger.error("Jersey classifier checkpoint unreadable: %s", exc)
                self.model = None
                return

        if isinstance(ckpt, dict):
            if "class_names" in ckpt:
                self.class_names = list(ckpt["class_names"])
            else:
                meta_path = resolve_jersey_meta(path)
                if meta_path is not None:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    self.class_names = list(meta.get("class_names", self.class_names))
                else:
                    logger.error(
                        "Jersey checkpoint missing class_names and %s.json",
                        path.stem,
                    )
                    self.model = None
                    return
            self.model_name = str(ckpt.get("model", self.model_name))
            self.input_size = int(ckpt.get("input_size", self.input_size))
            self._transform = self._make_transform(self.input_size)
            state = ckpt.get("model_state_dict", ckpt)
        else:
            state = ckpt

        try:
            model = _build_backbone(self.model_name, len(self.class_names))
            model.load_state_dict(state, strict=True)
            model.to(self.device)
            model.eval()
            self.model = model
        except Exception as exc:
            logger.error("Jersey classifier load failed: %s", exc)
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict_crop(self, crop_bgr: NDArray) -> tuple[int | None, float, str]:
        if self.model is None or crop_bgr.size == 0:
            return None, 0.0, "unknown"

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = float(probs.max().item()), int(probs.argmax().item())

        label = self.class_names[idx] if idx < len(self.class_names) else "unknown"
        if label == "unknown":
            return None, conf, label
        try:
            return int(label), conf, label
        except ValueError:
            return None, conf, label

    def _target_probability(self, crop_bgr: NDArray, target_number: int) -> float:
        if self.model is None or crop_bgr.size == 0:
            return 0.0
        label = str(target_number)
        if label not in self.class_names:
            return 0.0
        idx = self.class_names.index(label)
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
        return float(probs[idx].item())

    def _classifier_vote_reading(
        self,
        raw_crop: NDArray,
        *,
        target_number: int | None,
        num: int | None,
        conf: float,
    ) -> tuple[int | None, float]:
        """Classifier vote for lock — only when argmax matches the target jersey."""
        if self.model is None or raw_crop.size == 0:
            return None, 0.0
        if num is None or conf < _CLASSIFIER_MIN_CONF:
            return None, 0.0
        if target_number is not None and num != target_number:
            return None, 0.0
        if target_number is not None:
            target_prob = self._target_probability(raw_crop, target_number)
            return num, max(conf, target_prob)
        return num, conf

    def _pick_display_reading(
        self,
        raw_crop: NDArray,
        *,
        target_number: int | None,
        num: int | None,
        conf: float,
        label: str,
        ocr_num: int | None,
        ocr_conf: float,
        ocr_raw: list[tuple[str, float]],
    ) -> tuple[int | None, float, list[tuple[str, float]]]:
        if num is not None and conf >= _MIN_CONF:
            return num, conf, [(f"classifier:{label}", conf)]

        target_threshold = max(_TARGET_MIN_CONF, _NUMBER_MIN_CONF)
        if target_number is not None:
            target_prob = self._target_probability(raw_crop, target_number)
            if target_prob >= target_threshold and (
                num is None or num == target_number or target_prob > conf
            ):
                return target_number, target_prob, [
                    (f"classifier_target:{target_number}", target_prob)
                ]

        if num is not None and conf >= _SOFT_MIN_CONF:
            return num, conf, [(f"classifier_soft:{label}", conf)]

        if ocr_num is not None and ocr_conf >= _NUMBER_MIN_CONF:
            return ocr_num, ocr_conf, ocr_raw
        if num is not None and conf >= _SOFT_MIN_CONF:
            return num, conf, [(f"classifier_soft:{label}", conf)]
        if ocr_num is not None:
            return ocr_num, ocr_conf, ocr_raw
        return None, 0.0, ocr_raw

    def read_number_sources_detailed(
        self,
        frame: NDArray,
        bbox: list[float],
        *,
        target_number: int | None = None,
    ) -> tuple[
        int | None,
        float,
        int | None,
        float,
        int | None,
        float,
        list[tuple[str, float]],
    ]:
        """Display read plus separate classifier/OCR sources for weighted lock votes."""
        raw_crop = number_crop(frame, bbox)
        if raw_crop.size == 0:
            return None, 0.0, None, 0.0, None, 0.0, []

        num, conf, label = self.predict_crop(raw_crop)
        clf_num, clf_conf = self._classifier_vote_reading(
            raw_crop, target_number=target_number, num=num, conf=conf
        )
        ocr_crop = preprocess_crop_for_ocr(raw_crop)
        ocr_num, ocr_conf, ocr_raw = self._ocr_fallback.read(ocr_crop)
        display_num, display_conf, display_raw = self._pick_display_reading(
            raw_crop,
            target_number=target_number,
            num=num,
            conf=conf,
            label=label,
            ocr_num=ocr_num,
            ocr_conf=ocr_conf,
            ocr_raw=ocr_raw,
        )
        raw = list(display_raw)
        if clf_num is not None:
            raw.insert(0, (f"classifier_vote:{clf_num}", clf_conf))
        return (
            display_num,
            display_conf,
            clf_num,
            clf_conf,
            ocr_num,
            ocr_conf,
            raw,
        )

    def read_number_detailed(
        self,
        frame: NDArray,
        bbox: list[float],
        *,
        target_number: int | None = None,
    ) -> tuple[int | None, float, list[tuple[str, float]]]:
        """Returns (number, confidence, raw reads for debug)."""
        display_num, display_conf, _cn, _cc, _on, _oc, raw = (
            self.read_number_sources_detailed(frame, bbox, target_number=target_number)
        )
        return display_num, display_conf, raw

    def read_number(self, frame: NDArray, bbox: list[float]) -> tuple[int | None, float]:
        num, conf, _raw = self.read_number_detailed(frame, bbox)
        return num, conf


_classifier_instance: JerseyNumberClassifier | None = None


def get_jersey_classifier() -> JerseyNumberClassifier:
    """Process-wide lazy singleton (avoids reloading weights per analyze request)."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = JerseyNumberClassifier()
    return _classifier_instance
