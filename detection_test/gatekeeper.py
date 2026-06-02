from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bbox_utils import is_valid_person_bbox, normalize_bbox
from color_score import kit_histogram_match_score, kit_torso_primary_match

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class GatekeeperConfig:
    kit_hist_gate: float = 0.35
    kit_pixel_gate: float = 0.12
    kit_pixel_tolerance: int = 40
    skip_kit_gate: bool = False


@dataclass
class GatekeeperResult:
    passed: bool
    geometry_ok: bool
    kit_ok: bool
    kit_score: float
    bbox: list[float]


class Gatekeeper:
    """Stage 1: reject bad geometry and kit mismatch before jersey classifier runs."""

    def __init__(self, config: GatekeeperConfig | None = None) -> None:
        self.config = config or GatekeeperConfig()

    def evaluate(
        self,
        frame: NDArray,
        bbox: list[float],
        primary_hex: str,
        secondary_hex: str,
    ) -> GatekeeperResult:
        normalized = normalize_bbox(bbox, frame.shape)
        geometry_ok = is_valid_person_bbox(normalized, frame.shape)
        kit_score = 0.0
        kit_pixel = 0.0
        kit_ok = False
        if geometry_ok and not self.config.skip_kit_gate:
            kit_score = kit_histogram_match_score(
                frame,
                normalized,
                primary_hex,
                secondary_hex,
            )
            kit_pixel = kit_torso_primary_match(
                frame,
                normalized,
                primary_hex,
                tolerance=self.config.kit_pixel_tolerance,
            )
            kit_ok = (
                kit_score >= self.config.kit_hist_gate
                or kit_pixel >= self.config.kit_pixel_gate
            )
        elif geometry_ok:
            kit_ok = True
        return GatekeeperResult(
            passed=geometry_ok and kit_ok,
            geometry_ok=geometry_ok,
            kit_ok=kit_ok,
            kit_score=kit_score,
            bbox=normalized,
        )
