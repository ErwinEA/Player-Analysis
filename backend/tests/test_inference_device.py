"""Device resolution for MPS-first inference."""

from __future__ import annotations

import os

import pytest


def test_sam_device_candidates_mps_no_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAM_ALLOW_CPU_FALLBACK", raising=False)
    from backend.app.pipeline import segmentation as seg

    assert seg._device_candidates("mps") == ["mps"]
    assert seg._device_candidates("cuda") == ["cuda"]


def test_sam_device_candidates_allow_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAM_ALLOW_CPU_FALLBACK", "1")
    from backend.app.pipeline import segmentation as seg

    assert seg._device_candidates("mps") == ["mps", "cpu"]


def test_resolve_ultralytics_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_DEVICE", "mps")
    monkeypatch.delenv("YOLO_DEVICE", raising=False)
    from backend.app.pipeline.inference_device import resolve_ultralytics_device

    assert resolve_ultralytics_device() == "mps"


def test_yolo_device_overrides_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_DEVICE", "mps")
    monkeypatch.setenv("YOLO_DEVICE", "cpu")
    from backend.app.pipeline.inference_device import resolve_ultralytics_device

    assert resolve_ultralytics_device() == "cpu"
