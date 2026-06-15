#!/usr/bin/env bash
# Start the analyze API with the same env as the validated CLI E2E runs (MPS + BoT-SORT + frame cap).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source backend/.venv/bin/activate

export SAM_DEVICE="${SAM_DEVICE:-mps}"
export JERSEY_CLS_DEVICE="${JERSEY_CLS_DEVICE:-mps}"
export INFERENCE_DEVICE="${INFERENCE_DEVICE:-mps}"
export MAX_FRAMES="${MAX_FRAMES:-1500}"
export TRACKER_TYPE="${TRACKER_TYPE:-botsort}"
export TRACK_BUFFER="${TRACK_BUFFER:-120}"
export REID_FALLBACK_MAX_CENTER_FRAC="${REID_FALLBACK_MAX_CENTER_FRAC:-0.6}"
export REID_FALLBACK_THRESH="${REID_FALLBACK_THRESH:-0.55}"
export REID_FALLBACK_MARGIN="${REID_FALLBACK_MARGIN:-0.05}"
export POSSESS_RADIUS_M="${POSSESS_RADIUS_M:-2.0}"
export OCR_EVERY="${OCR_EVERY:-5}"
# SAM runs every Nth frame (applies to all sports; raise for faster dev runs)
export SAM_STRIDE="${SAM_STRIDE:-1}"
export BALL_EVENTS_FRAME_LOG="${BALL_EVENTS_FRAME_LOG:-0}"

echo "SAM_DEVICE=$SAM_DEVICE JERSEY_CLS_DEVICE=$JERSEY_CLS_DEVICE INFERENCE_DEVICE=$INFERENCE_DEVICE MAX_FRAMES=$MAX_FRAMES SAM_STRIDE=$SAM_STRIDE TRACKER_TYPE=$TRACKER_TYPE"
# For LAN frontend access, set UVICORN_HOST=0.0.0.0 (CORS already allows 192.168/10.*).
export UVICORN_HOST="${UVICORN_HOST:-127.0.0.1}"
exec uvicorn backend.app.main:app --reload --host "$UVICORN_HOST" --port 8000 --log-level info
