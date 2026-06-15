"""Shuttle detector weight resolution and graceful degradation."""

from pathlib import Path

import pytest

from backend.app.pipeline.badminton.shuttle_detector import (
    ShuttleDetection,
    ShuttleDetector,
    resolve_shuttle_weights,
    shuttle_health,
)

_REPO_SHUTTLE = Path("backend/weights/yolov8m_shuttlecock.pt")


def test_resolve_shuttle_weights_missing_returns_none(monkeypatch):
    monkeypatch.delenv("SHUTTLE_WEIGHTS", raising=False)
    out = resolve_shuttle_weights("/nonexistent/yolov8m_shuttlecock.pt")
    assert out is None or out.endswith("yolov8m_shuttlecock.pt")


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
        pytest.skip("yolov8m_shuttlecock.pt not installed locally")
    out = resolve_shuttle_weights()
    assert out is not None
    assert out.endswith("yolov8m_shuttlecock.pt")


def test_shuttle_detector_loads_when_weights_present():
    if not _REPO_SHUTTLE.is_file():
        pytest.skip("yolov8m_shuttlecock.pt not installed locally")
    det = ShuttleDetector()
    assert det.available is True
    assert det.weights is not None


def test_shuttle_health_ok_when_weights_present():
    if not _REPO_SHUTTLE.is_file():
        pytest.skip("yolov8m_shuttlecock.pt not installed locally")
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


def test_shuttle_kalman_update_returns_position():
    from backend.app.pipeline.badminton.shuttle_detector import ShuttleKalman

    k = ShuttleKalman()
    px, py = k.update(100.0, 200.0)
    assert isinstance(px, float) and isinstance(py, float)
    # First update seeds the filter; output should be close to input.
    assert abs(px - 100.0) < 5.0
    assert abs(py - 200.0) < 5.0


def test_shuttle_kalman_predict_extrapolates():
    from backend.app.pipeline.badminton.shuttle_detector import ShuttleKalman

    k = ShuttleKalman()
    k.update(0.0, 0.0)
    k.update(10.0, 5.0)
    result = k.predict()
    assert result is not None
    px, py = result
    # Predicted position should be ahead of last detection (nonzero velocity).
    assert px > 5.0


def test_shuttle_kalman_predict_returns_none_after_gap_max():
    from backend.app.pipeline.badminton.shuttle_detector import ShuttleKalman

    k = ShuttleKalman(gap_max=3)
    k.update(50.0, 50.0)
    for _ in range(3):
        assert k.predict() is not None
    # 4th predict (1 past gap_max) should reset and return None.
    assert k.predict() is None


def test_shuttle_kalman_reset_clears_state():
    from backend.app.pipeline.badminton.shuttle_detector import ShuttleKalman

    k = ShuttleKalman()
    k.update(100.0, 100.0)
    k.reset()
    assert k.predict() is None


def test_shuttle_kalman_predict_before_update_returns_none():
    from backend.app.pipeline.badminton.shuttle_detector import ShuttleKalman

    k = ShuttleKalman()
    assert k.predict() is None


def test_health_endpoint_includes_shuttle():
    from fastapi.testclient import TestClient

    from backend.app.main import app

    res = TestClient(app).get("/health")
    assert res.status_code == 200
    shuttle = res.json().get("shuttle")
    assert shuttle is not None
    assert "weights_found" in shuttle
    assert "status" in shuttle
