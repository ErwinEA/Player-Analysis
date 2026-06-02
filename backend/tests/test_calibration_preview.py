"""Pitch calibration preview API tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas import PitchCalibrationSaveRequest


@pytest.fixture()
def client():
    return TestClient(app)


def _wide_boundary(w: int, h: int, n: int = 6) -> list[list[float]]:
    if n >= 10:
        return [
            [w * 0.1, h * 0.2],
            [w * 0.35, h * 0.18],
            [w * 0.5, h * 0.17],
            [w * 0.9, h * 0.2],
            [w * 0.92, h * 0.45],
            [w * 0.93, h * 0.7],
            [w * 0.95, h * 0.95],
            [w * 0.7, h * 0.97],
            [w * 0.3, h * 0.96],
            [w * 0.05, h * 0.95],
        ]
    return [
        [w * 0.1, h * 0.2],
        [w * 0.5, h * 0.17],
        [w * 0.9, h * 0.2],
        [w * 0.95, h * 0.95],
        [w * 0.7, h * 0.97],
        [w * 0.05, h * 0.95],
    ][:n]


def test_preview_returns_confidence(client):
    res = client.get("/api/pitch/frame?name=testmatch2&frame=100")
    if res.status_code == 404:
        pytest.skip("Default video not installed")
    frame = res.json()
    w, h = frame["width"], frame["height"]
    preview = client.post(
        "/api/pitch/calibration/preview",
        json={
            "name": "testmatch2",
            "frame_index": 100,
            "image_boundary_points": _wide_boundary(w, h, 6),
        },
    )
    assert preview.status_code == 200
    body = preview.json()
    assert body["confidence"] >= 0.8
    assert body["probe_count"] >= 3
    assert len(body["fitted_quad"]) == 4
    assert body["mode"] == "outline"


def test_save_with_six_points_includes_confidence(client):
    res = client.get("/api/pitch/frame?name=testmatch2&frame=100")
    if res.status_code == 404:
        pytest.skip("Default video not installed")
    frame = res.json()
    w, h = frame["width"], frame["height"]
    save = client.post(
        "/api/pitch/calibration",
        json={
            "name": "testmatch2",
            "frame_index": 100,
            "image_boundary_points": _wide_boundary(w, h, 6),
        },
    )
    assert save.status_code == 200
    body = save.json()
    assert body.get("confidence") is not None
    assert body["confidence"] >= 0.8


def test_preview_with_client_image_size(client):
    preview = client.post(
        "/api/pitch/calibration/preview",
        json={
            "name": "upload_preview_only",
            "frame_index": 0,
            "image_width": 1280,
            "image_height": 720,
            "image_boundary_points": _wide_boundary(1280, 720, 6),
        },
    )
    assert preview.status_code == 200
    body = preview.json()
    assert len(body["fitted_quad"]) == 4
    assert body["probe_total"] >= 1


def test_image_size_must_be_paired():
    with pytest.raises(ValueError):
        PitchCalibrationSaveRequest(
            name="x",
            frame_index=0,
            image_width=1280,
            image_boundary_points=_wide_boundary(1280, 720, 6),
        )


def test_three_points_rejected_by_schema():
    with pytest.raises(ValueError):
        PitchCalibrationSaveRequest(
            name="x",
            frame_index=0,
            image_boundary_points=[[0, 0], [1, 1], [2, 2]],
        )


def test_preview_three_points_422(client):
    res = client.post(
        "/api/pitch/calibration/preview",
        json={
            "name": "testmatch2",
            "frame_index": 100,
            "image_boundary_points": [[0, 0], [1, 1], [2, 2]],
        },
    )
    assert res.status_code == 422
