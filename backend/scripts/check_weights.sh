#!/usr/bin/env bash
# Check presence of optional ML weights and print fetch steps.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEIGHTS="$ROOT/weights"

check() {
  local name="$1"
  local path="$2"
  local hint="$3"
  if [[ -f "$path" ]]; then
    echo "OK   $name"
  else
    echo "MISS $name — $hint"
  fi
}

echo "Weights directory: $WEIGHTS"
check "yolov8m_shuttlecock.pt" "$WEIGHTS/yolov8m_shuttlecock.pt" \
  "badminton shuttle detection; see README.md"
check "yolov8n_ball.pt" "$WEIGHTS/yolov8n_ball.pt" \
  "football ball events; see README.md"
check "jersey_number_b0.pt" "$WEIGHTS/jersey_number_b0.pt" \
  "optional jersey OCR; see README.md"
check "osnet_x1_0_soccernet.pth" "$WEIGHTS/osnet_x1_0_soccernet.pth" \
  "ReID fallback; see README.md"
check "mobile_sam.pt" "$WEIGHTS/mobile_sam.pt" \
  "MobileSAM segmentation; run backend/scripts/install_mobile_sam.sh"

echo ""
echo "Env toggles: SHUTTLE_MAX_SPEED_PX, SHUTTLE_COURT_MARGIN_M, YOLO_IMGSZ, SAM_ENABLED (default off for badminton)."
