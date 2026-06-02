from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from bbox_utils import crop_region
from ocr_profile import JerseyNameOCR, number_crop, preprocess_crop
from weights_config import JERSEY_NUMBER_META, resolve_jersey_weights

if TYPE_CHECKING:
    from numpy.typing import NDArray

_MIN_CONF = float(os.environ.get("JERSEY_CLS_MIN_CONF", "0.75"))
_TARGET_MIN_CONF = float(os.environ.get("JERSEY_TARGET_MIN_CONF", "0.25"))
_SOFT_MIN_CONF = float(os.environ.get("JERSEY_SOFT_MIN_CONF", "0.25"))
_DEFAULT_INPUT_SIZE = 224
_DEFAULT_CLASS_NAMES = ["unknown"] + [str(n) for n in range(1, 100)]


def _resolve_inference_device() -> torch.device:
    pref = os.environ.get("JERSEY_CLS_DEVICE", "auto").lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "cuda":
        return torch.device("cpu")
    return torch.device("cpu")


class _DigitOCRFallback:
    """EasyOCR digits-only fallback when classifier is unavailable or low confidence."""

    def __init__(self) -> None:
        self._reader = None

    def _ensure_reader(self):
        if self._reader is None:
            import easyocr

            use_gpu = os.environ.get("OCR_GPU", "0") == "1"
            self._reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        return self._reader

    def read(self, crop: NDArray) -> tuple[int | None, float]:
        if crop.size == 0:
            return None, 0.0
        try:
            results = self._ensure_reader().readtext(
                crop, detail=1, paragraph=False, allowlist="0123456789"
            )
        except Exception:
            return None, 0.0
        best_num: int | None = None
        best_conf = 0.0
        for item in results or []:
            if not item or len(item) < 3:
                continue
            digits = re.sub(r"\D", "", str(item[1]))
            if 1 <= len(digits) <= 2:
                try:
                    num = int(digits)
                except ValueError:
                    continue
                conf = float(item[2])
                if conf > best_conf:
                    best_num = num
                    best_conf = conf
        return best_num, best_conf


def _build_backbone(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


class JerseyNumberClassifier:
    """EfficientNet-B0 jersey number classifier (1–99 + unknown) with EasyOCR fallback."""

    def __init__(self, weights_path: str | None = None) -> None:
        self.weights_path = resolve_jersey_weights(weights_path)
        self.class_names: list[str] = list(_DEFAULT_CLASS_NAMES)
        self.model_name = "efficientnet_b0"
        self.input_size = _DEFAULT_INPUT_SIZE
        self.model: nn.Module | None = None
        self.device = _resolve_inference_device()
        self._ocr_fallback = _DigitOCRFallback()
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((_DEFAULT_INPUT_SIZE, _DEFAULT_INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        if self.weights_path is not None:
            self._load_weights(self.weights_path)
        elif self.weights_path is None:
            import logging

            logging.getLogger(__name__).warning(
                "Jersey classifier weights not found — OCR fallback only."
            )

    def _set_transform(self, input_size: int) -> None:
        self.input_size = input_size
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_weights(self, path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            if "class_names" in ckpt:
                self.class_names = list(ckpt["class_names"])
            elif JERSEY_NUMBER_META.is_file():
                meta = json.loads(JERSEY_NUMBER_META.read_text(encoding="utf-8"))
                self.class_names = list(meta.get("class_names", self.class_names))
            self.model_name = str(ckpt.get("model", self.model_name))
            self._set_transform(int(ckpt.get("input_size", self.input_size)))
            state = ckpt.get("model_state_dict", ckpt)
        else:
            state = ckpt

        model = _build_backbone(self.model_name, len(self.class_names))
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.model = model

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict_crop(self, crop_bgr: NDArray) -> tuple[int | None, float, str]:
        """Classify a BGR jersey-back crop. Returns (number, confidence, label)."""
        if self.model is None or crop_bgr.size == 0:
            return None, 0.0, "unknown"

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = float(probs.max().item()), int(probs.argmax().item())

        label = self.class_names[idx] if idx < len(self.class_names) else "unknown"
        if label == "unknown" or label.lower() == "unknown":
            return None, conf, label
        try:
            return int(label), conf, label
        except ValueError:
            return None, conf, label

    def read_number(self, frame: NDArray, bbox: list[float]) -> tuple[int | None, float]:
        num, conf, _source = self.read_number_detailed(frame, bbox)
        return num, conf

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

    def read_number_detailed(
        self,
        frame: NDArray,
        bbox: list[float],
        *,
        target_number: int | None = None,
    ) -> tuple[int | None, float, str]:
        raw_crop = number_crop(frame, bbox)
        if raw_crop.size == 0:
            return None, 0.0, "none"

        num, conf, label = self.predict_crop(raw_crop)
        if num is not None and label != "unknown" and conf >= _MIN_CONF:
            return num, conf, "classifier"

        if target_number is not None:
            target_prob = self._target_probability(raw_crop, target_number)
            if target_prob >= _TARGET_MIN_CONF:
                return target_number, target_prob, "classifier_target"

        if num is not None and conf >= _SOFT_MIN_CONF:
            return num, conf, "classifier_soft"

        ocr_crop = preprocess_crop(raw_crop)
        ocr_num, ocr_conf = self._ocr_fallback.read(ocr_crop)
        ocr_min = float(os.environ.get("NUMBER_MIN_CONF", "0.4"))
        if ocr_num is not None and ocr_conf >= ocr_min:
            return ocr_num, ocr_conf, "ocr"
        if num is not None and conf >= _SOFT_MIN_CONF:
            return num, conf, "classifier_soft"
        if ocr_num is not None:
            return ocr_num, ocr_conf, "ocr_low"
        return None, 0.0, "none"


class JerseyProfileReader:
    """Hybrid jersey number (classifier + OCR) and name (EasyOCR) reader."""

    def __init__(self) -> None:
        self.number_classifier = JerseyNumberClassifier()
        self.name_ocr = JerseyNameOCR()

    def read_number(self, frame: NDArray, bbox: list[float]) -> tuple[int | None, float]:
        return self.number_classifier.read_number(frame, bbox)

    def read_number_detailed(
        self,
        frame: NDArray,
        bbox: list[float],
        *,
        target_number: int | None = None,
    ) -> tuple[int | None, float, str]:
        return self.number_classifier.read_number_detailed(
            frame, bbox, target_number=target_number
        )

    def read_name(self, frame: NDArray, bbox: list[float]) -> tuple[str | None, float]:
        return self.name_ocr.read_name(frame, bbox)

    @staticmethod
    def name_crop(frame: NDArray, bbox: list[float]) -> NDArray:
        return JerseyNameOCR.name_crop(frame, bbox)

    @staticmethod
    def number_crop(frame: NDArray, bbox: list[float]) -> NDArray:
        return number_crop(frame, bbox)
