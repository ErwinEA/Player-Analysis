"""Tests for gatekeeper stage-1 filters."""

from __future__ import annotations

import numpy as np

from gatekeeper import Gatekeeper, GatekeeperConfig


def test_rejects_giant_offscreen_bbox():
    gate = Gatekeeper(GatekeeperConfig(kit_hist_gate=0.0))
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    bbox = [-68.22, 147.78, 682.67, 1907.54]
    result = gate.evaluate(frame, bbox, "#ffffff", "#000000")
    assert not result.passed
    assert not result.geometry_ok


def test_kit_gate_blocks_when_histogram_low():
    gate = Gatekeeper(GatekeeperConfig(kit_hist_gate=0.99))
    frame = np.full((400, 200, 3), (0, 128, 255), dtype=np.uint8)
    bbox = [50.0, 50.0, 150.0, 350.0]
    result = gate.evaluate(frame, bbox, "#ff0000", "#00ff00")
    assert result.geometry_ok
    assert not result.kit_ok
    assert not result.passed
