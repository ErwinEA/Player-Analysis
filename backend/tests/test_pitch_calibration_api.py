"""Pitch frame and calibration save API tests."""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture()
def client():
    return TestClient(app)


def test_pitch_frame_default_video(client):
    res = client.get("/api/pitch/frame?name=testmatch2&frame=100")
    if res.status_code == 404:
        pytest.skip("Default video not installed on this machine")
    assert res.status_code == 200
    body = res.json()
    assert body["frame_index"] == 100
    assert len(body["image_jpeg_base64"]) > 100
    raw = base64.b64decode(body["image_jpeg_base64"])
    assert raw[:2] == b"\xff\xd8"


def test_save_pitch_calibration(client):
    res = client.get("/api/pitch/frame?name=testmatch2&frame=100")
    if res.status_code == 404:
        pytest.skip("Default video not installed")
    frame = res.json()
    w, h = frame["width"], frame["height"]
    corners = [
        [w * 0.1, h * 0.2],
        [w * 0.9, h * 0.2],
        [w * 0.95, h * 0.95],
        [w * 0.05, h * 0.95],
    ]
    save = client.post(
        "/api/pitch/calibration",
        json={
            "name": "testmatch2",
            "frame_index": 100,
            "image_corners": corners,
        },
    )
    assert save.status_code == 200
    body = save.json()
    assert "template_url" in body
    assert body.get("confidence") is not None


def test_save_pitch_calibration_boundary_points(client):
    res = client.get("/api/pitch/frame?name=testmatch2&frame=100")
    if res.status_code == 404:
        pytest.skip("Default video not installed")
    frame = res.json()
    w, h = frame["width"], frame["height"]
    # Clockwise outline (matches production UI); corners at indices 0,3,6,9
    boundary = [
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
    save = client.post(
        "/api/pitch/calibration",
        json={
            "name": "testmatch2",
            "frame_index": 100,
            "image_boundary_points": boundary,
        },
    )
    assert save.status_code == 200
    assert "template_url" in save.json()
