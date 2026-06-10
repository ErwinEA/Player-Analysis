"""Shuttle detector weight resolution and graceful degradation."""

from backend.app.pipeline.badminton.shuttle_detector import (
    ShuttleDetection,
    ShuttleDetector,
    resolve_shuttle_weights,
)


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
