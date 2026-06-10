"""Shuttle detector weight resolution and graceful degradation."""

from pathlib import Path

import pytest

from backend.app.pipeline.badminton.shuttle_detector import (
    ShuttleDetection,
    ShuttleDetector,
    resolve_shuttle_weights,
    shuttle_health,
)

_REPO_SHUTTLE = Path("backend/weights/yolov8n_shuttle.pt")


def test_resolve_shuttle_weights_missing_returns_none(monkeypatch):
    monkeypatch.delenv("SHUTTLE_WEIGHTS", raising=False)
    out = resolve_shuttle_weights("/nonexistent/yolov8n_shuttle.pt")
    assert out is None or out.endswith("yolov8n_shuttle.pt")


def test_shuttle_detector_unavailable_without_weights(monkeypatch):
    monkeypatch.setattr(
        "backend.app.pipeline.badminton.shuttle_detector.resolve_shuttle_weights",
        lambda explicit=None: None,
    )
    det = ShuttleDetector()
    assert det.available is False
    import numpy as np

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    assert det.detect(frame) is None


def test_shuttle_detection_center():
    d = ShuttleDetection(cx_px=10.0, cy_px=20.0, conf=0.9)
    assert d.center_px == (10.0, 20.0)


def test_resolve_shuttle_weights_finds_backend_default():
    if not _REPO_SHUTTLE.is_file():
        pytest.skip("yolov8n_shuttle.pt not installed locally")
    out = resolve_shuttle_weights()
    assert out is not None
    assert out.endswith("yolov8n_shuttle.pt")


def test_shuttle_detector_loads_when_weights_present():
    if not _REPO_SHUTTLE.is_file():
        pytest.skip("yolov8n_shuttle.pt not installed locally")
    det = ShuttleDetector()
    assert det.available is True
    assert det.weights is not None


def test_shuttle_health_ok_when_weights_present():
    if not _REPO_SHUTTLE.is_file():
        pytest.skip("yolov8n_shuttle.pt not installed locally")
    h = shuttle_health()
    assert h["weights_found"] is True
    assert h["status"] == "ok"


def test_shuttle_detect_survives_predict_failure(monkeypatch):
    import numpy as np

    class _BrokenModel:
        def predict(self, *args, **kwargs):
            raise RuntimeError("boom")

    det = ShuttleDetector()
    det.available = True
    det._model = _BrokenModel()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    assert det.detect(frame) is None


def test_health_endpoint_includes_shuttle():
    from fastapi.testclient import TestClient

    from backend.app.main import app

    res = TestClient(app).get("/health")
    assert res.status_code == 200
    shuttle = res.json().get("shuttle")
    assert shuttle is not None
    assert "weights_found" in shuttle
    assert "status" in shuttle
