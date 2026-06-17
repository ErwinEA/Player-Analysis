"""Production model checkpoint paths under backend/weights/."""

from __future__ import annotations

from pathlib import Path

_PIPELINE = Path(__file__).resolve().parent
BACKEND_WEIGHTS = _PIPELINE.parent / "weights"

OSNET_CHECKPOINT = "osnet_x1_0_soccernet.pth"
OSNET_ALIAS = "osnet_soccer_reid.pth"
JERSEY_CHECKPOINT = "jersey_number_b0.pt"
BALL_CHECKPOINT = "yolov8n_ball.pt"
SAM_CHECKPOINT = "mobile_sam.pt"
