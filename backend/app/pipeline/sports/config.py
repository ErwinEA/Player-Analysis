"""Court dimensions per sport."""

from __future__ import annotations

COURT_CONFIG: dict[str, dict[str, float]] = {
    "football": {"length_m": 105.0, "width_m": 68.0},
    "badminton": {"length_m": 13.4, "width_m": 6.1},
}

_DEFAULT_SPORT = "football"
_DIM_TOLERANCE_M = 0.1


def court_dimensions(sport: str) -> tuple[float, float]:
    """Return (length_m, width_m) for the given sport."""
    key = sport.strip().lower() if sport else _DEFAULT_SPORT
    cfg = COURT_CONFIG.get(key)
    if cfg is None:
        cfg = COURT_CONFIG[_DEFAULT_SPORT]
    return float(cfg["length_m"]), float(cfg["width_m"])


def calibration_matches_sport(
    *,
    sport: str,
    pitch_length_m: float,
    pitch_width_m: float,
    tolerance_m: float = _DIM_TOLERANCE_M,
) -> bool:
    """True when stored calibration dimensions match the sport's court config."""
    expected_len, expected_wid = court_dimensions(sport)
    return (
        abs(pitch_length_m - expected_len) <= tolerance_m
        and abs(pitch_width_m - expected_wid) <= tolerance_m
    )
