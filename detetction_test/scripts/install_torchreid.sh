#!/usr/bin/env bash
# Install KaiyangZhou/deep-person-reid (torchreid) on Python 3.10+.
# Plain `pip install git+...` often fails: setup.py needs numpy at build time
# and its version lookup breaks with modern setuptools.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/.vendor/deep-person-reid"

cd "$ROOT"
if [[ ! -d test/bin ]]; then
  echo "Create the venv first: python3 -m venv test && source test/bin/activate"
  exit 1
fi
# shellcheck disable=SC1091
source test/bin/activate

pip install -q numpy cython gdown

if [[ ! -d "$VENDOR/.git" ]]; then
  git clone --depth 1 https://github.com/KaiyangZhou/deep-person-reid.git "$VENDOR"
fi

# Patch broken find_version() (KeyError: __version__) when setup imports torchreid.
python - <<'PY'
from pathlib import Path
setup = Path(".vendor/deep-person-reid/setup.py")
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

pip install -e "$VENDOR" --no-build-isolation
python -c "from torchreid.utils import FeatureExtractor; print('torchreid installed OK')"
