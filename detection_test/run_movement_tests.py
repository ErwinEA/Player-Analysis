#!/usr/bin/env python3
"""Run movement_stats unit tests (avoids generic ``python -c`` import pitfalls)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests import test_movement_stats as t


def main() -> int:
    failed = 0
    for name in sorted(n for n in dir(t) if n.startswith("test_")):
        try:
            getattr(t, name)()
            print(f"PASS {name}")
        except Exception as exc:
            print(f"FAIL {name}: {exc}")
            failed += 1
    if failed:
        print(f"\n{failed} failed")
        return 1
    print(f"\nAll {len([n for n in dir(t) if n.startswith('test_')])} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
