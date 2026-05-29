#!/usr/bin/env bash
# Install MobileSAM + timm into backend/.venv (player mask overlay).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ ! -d "$ROOT/.venv" ]]; then
  echo "Create backend venv first: cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

"$ROOT/.venv/bin/pip" install -q "timm>=0.9.0" \
  "git+https://github.com/ChaoningZhang/MobileSAM.git"

WEIGHTS="$ROOT/weights/mobile_sam.pt"
if [[ ! -f "$WEIGHTS" ]]; then
  echo "Downloading mobile_sam.pt to $WEIGHTS …"
  curl -fsSL -o "$WEIGHTS" \
    "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
fi

"$ROOT/.venv/bin/python" -c "
from backend.app.pipeline.segmentation import get_mobile_sam_segmenter, prepare_mobile_sam_for_run
prepare_mobile_sam_for_run()
seg = get_mobile_sam_segmenter()
assert seg is not None, 'MobileSAM failed to load'
print('MobileSAM OK in backend venv')
"
