#!/usr/bin/env python3
"""Install default demo video into backend/data/videos/."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from backend.app.video_store import default_video_path, ensure_default_video_installed


def main() -> None:
    path = ensure_default_video_installed()
    print(f"Default video ready: {path}")


if __name__ == "__main__":
    main()
