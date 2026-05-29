"""Lightweight terminal UI for the detection pipeline (no extra dependencies)."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass


class _Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"


def _supports_color() -> bool:
    return sys.stdout.isatty() and sys.platform != "win32"


def _c(text: str, code: str) -> str:
    if not _supports_color():
        return text
    return f"{code}{text}{_Ansi.RESET}"


@dataclass
class PipelineUI:
    """Friendly step-by-step CLI for new users."""

    total_steps: int = 4

    def banner(self) -> None:
        line = "═" * 52
        print()
        print(_c(line, _Ansi.CYAN))
        print(_c("  Player Analysis Pipeline", _Ansi.BOLD + _Ansi.CYAN))
        print(_c("  YOLO → Filter → ByteTrack → Jersey → JSON", _Ansi.DIM))
        print(_c(line, _Ansi.CYAN))
        print()

    def step_header(self, index: int, title: str, detail: str = "") -> None:
        print(_c(f"Step {index}/{self.total_steps}  {title}", _Ansi.BOLD + _Ansi.BLUE))
        if detail:
            print(_c(f"  {detail}", _Ansi.DIM))
        print()

    def info(self, message: str) -> None:
        print(_c(f"  • {message}", _Ansi.DIM))

    def success(self, message: str) -> None:
        print(_c(f"  ✓ {message}", _Ansi.GREEN))

    def warn(self, message: str) -> None:
        print(_c(f"  ! {message}", _Ansi.YELLOW))

    def error(self, message: str) -> None:
        print(_c(f"  ✗ {message}", _Ansi.RED))

    def progress(self, current: int, total: int, *, prefix: str = "") -> None:
        if total <= 0:
            return
        width = 28
        filled = int(width * current / total)
        bar = "█" * filled + "░" * (width - filled)
        pct = 100 * current / total
        label = f"{prefix} " if prefix else ""
        line = f"  {label}[{bar}] {current}/{total} ({pct:.0f}%)"
        print(_c(line, _Ansi.MAGENTA), end="\r", flush=True)
        if current >= total:
            print()

    def stage_timing(self, label: str, seconds: float) -> None:
        print(_c(f"  ⏱ {label}: {seconds:.1f}s", _Ansi.DIM))

    def summary_box(self, lines: list[tuple[str, str]]) -> None:
        print()
        print(_c("  ┌─ Results " + "─" * 38, _Ansi.CYAN))
        for key, value in lines:
            print(_c(f"  │ {key:<22} {value}", _Ansi.CYAN))
        print(_c("  └" + "─" * 49, _Ansi.CYAN))
        print()

    def next_steps(self, paths: list[str]) -> None:
        print(_c("  What you can do next:", _Ansi.BOLD))
        for p in paths:
            print(_c(f"    → {p}", _Ansi.DIM))
        print()


class StageTimer:
    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start
