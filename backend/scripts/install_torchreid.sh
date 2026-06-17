#!/usr/bin/env bash
# Install torchreid (OSNet) into backend/.venv.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/.vendor/deep-person-reid"

if [[ ! -d "$ROOT/.venv" ]]; then
  echo "Create backend venv first: cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

pip_cmd="$ROOT/.venv/bin/pip"
python_cmd="$ROOT/.venv/bin/python"

"$pip_cmd" install -q numpy cython

if [[ ! -d "$VENDOR/.git" ]]; then
  git clone --depth 1 https://github.com/KaiyangZhou/deep-person-reid.git "$VENDOR"
fi

# Patch broken find_version() (KeyError: __version__) when setup imports torchreid.
"$python_cmd" - <<PY
from pathlib import Path
setup = Path("$VENDOR/setup.py")
text = setup.read_text()
old = """def find_version():
    version_file = 'torchreid/__init__.py'
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']"""
new = """def find_version():
    version_file = 'torchreid/__init__.py'
    with open(version_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=', 1)[1].strip().strip("'\\"")
    raise RuntimeError(f'Unable to find version in {version_file}')"""
if old in text:
    setup.write_text(text.replace(old, new))
PY

"$pip_cmd" install -e "$VENDOR" --no-build-isolation
"$python_cmd" -c "from torchreid.utils import FeatureExtractor; print('torchreid OK in backend venv')"
