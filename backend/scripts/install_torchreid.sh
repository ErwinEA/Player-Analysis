#!/usr/bin/env bash
# Install torchreid (OSNet) into backend/.venv from detection_test vendor tree.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$(cd "$ROOT/../detection_test" && pwd)/.vendor/deep-person-reid"

if [[ ! -d "$ROOT/.venv" ]]; then
  echo "Create backend venv first: cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

if [[ ! -d "$VENDOR" ]]; then
  echo "Missing $VENDOR — run detection_test/scripts/install_torchreid.sh first."
  exit 1
fi

"$ROOT/.venv/bin/pip" install -q numpy cython
"$ROOT/.venv/bin/pip" install -e "$VENDOR" --no-build-isolation
"$ROOT/.venv/bin/python" -c "from torchreid.utils import FeatureExtractor; print('torchreid OK in backend venv')"
