"""Jersey classifier + OCR fallback wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from backend.app.pipeline.jersey_number import (
    JerseyNumberClassifier,
    _parse_digits_from_ocr,
    get_jersey_classifier,
    preprocess_crop_for_ocr,
)
from backend.app.pipeline.jersey_weights import resolve_jersey_meta, resolve_jersey_weights
from backend.app.pipeline.ocr import JerseyOCR


def test_resolve_jersey_weights_env(monkeypatch, tmp_path):
    ckpt = tmp_path / "custom.pt"
    ckpt.write_bytes(b"x")
    monkeypatch.setenv("JERSEY_WEIGHTS", str(ckpt))
    assert resolve_jersey_weights() == ckpt


def test_resolve_jersey_meta_only_adjacent(tmp_path):
    ckpt = tmp_path / "model.pt"
    ckpt.touch()
    assert resolve_jersey_meta(ckpt) is None
    meta = tmp_path / "model.json"
    meta.write_text('{"class_names": ["unknown", "1"]}', encoding="utf-8")
    assert resolve_jersey_meta(ckpt) == meta


def test_parse_digits_rejects_zero_and_hundreds():
    results = [
        (None, "00", 0.9),
        (None, "100", 0.8),
        (None, "23", 0.7),
    ]
    num, conf, _ = _parse_digits_from_ocr(results)
    assert num == 23
    assert conf == 0.7


def test_jersey_ocr_uses_singleton_classifier():
    ocr_a = JerseyOCR()
    ocr_b = JerseyOCR()
    assert ocr_a._number_reader is ocr_b._number_reader


def test_get_jersey_classifier_singleton():
    import backend.app.pipeline.jersey_number as jn

    jn._classifier_instance = None
    first = get_jersey_classifier()
    second = get_jersey_classifier()
    assert first is second
    jn._classifier_instance = None


def test_classifier_vote_reading_requires_argmax_target_match():
    clf = JerseyNumberClassifier.__new__(JerseyNumberClassifier)
    clf.model = MagicMock()
    clf.class_names = ["unknown"] + [str(n) for n in range(1, 100)]
    crop = np.zeros((40, 30, 3), dtype=np.uint8)

    with patch.object(clf, "_target_probability", return_value=0.95):
        vote_num, vote_conf = clf._classifier_vote_reading(
            crop, target_number=23, num=14, conf=0.9
        )
        assert vote_num is None
        assert vote_conf == 0.0

        vote_num, vote_conf = clf._classifier_vote_reading(
            crop, target_number=23, num=23, conf=0.7
        )
        assert vote_num == 23
        assert vote_conf == 0.95


def test_read_number_detailed_target_requires_agreement():
    clf = JerseyNumberClassifier.__new__(JerseyNumberClassifier)
    clf.model = MagicMock()
    clf.device = "cpu"
    clf._ocr_fallback = MagicMock()
    clf._ocr_fallback.read.return_value = (None, 0.0, [])
    clf.class_names = ["unknown"] + [str(n) for n in range(1, 100)]

    frame = np.zeros((100, 80, 3), dtype=np.uint8)
    bbox = [10.0, 10.0, 70.0, 90.0]

    with (
        patch.object(clf, "predict_crop", return_value=(7, 0.9, "7")),
        patch.object(clf, "_target_probability", return_value=0.99),
    ):
        num, conf, raw = clf.read_number_detailed(
            frame, bbox, target_number=10
        )
        assert num == 7
        assert conf == 0.9
        assert any(entry[0].startswith("classifier") for entry in raw)


def test_preprocess_crop_empty():
    assert preprocess_crop_for_ocr(np.array([])).size == 0
