"""Per-track jersey labeling (Gatekeeper + classifier) over detections.json frames."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2

from bbox_utils import is_valid_person_bbox, normalize_bbox
from gatekeeper import Gatekeeper, GatekeeperConfig
from jersey_number import JerseyProfileReader


@dataclass
class LabelTracksConfig:
    jersey_every: int = 5
    kit_hist_gate: float = 0.35
    kit_pixel_gate: float = 0.12
    kit_pixel_tolerance: int = 40
    use_kit_gate: bool = True
    primary_color: str = "#1F3D31"
    secondary_color: str = "#000000"


@dataclass
class _TrackAgg:
    reads: list[tuple[int, int, float, str]] = field(default_factory=list)

    def add(self, frame: int, number: int | None, conf: float, source: str) -> None:
        if number is None:
            return
        self.reads.append((frame, number, conf, source))

    def best_number(self) -> tuple[int | None, float, str]:
        if not self.reads:
            return None, 0.0, "none"
        scores: dict[int, float] = defaultdict(float)
        sources: dict[int, str] = {}
        for _f, num, conf, src in self.reads:
            scores[num] += conf
            if num not in sources or conf > 0:
                sources[num] = src
        best = max(scores.items(), key=lambda kv: kv[1])
        return best[0], best[1] / max(1, sum(1 for r in self.reads if r[1] == best[0])), sources.get(best[0], "unknown")


def label_detections(
    payload: dict[str, Any],
    frames_dir: Path,
    reader: JerseyProfileReader,
    config: LabelTracksConfig,
    *,
    on_progress: Callable[[int, int], None] | None = None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Enrich detections payload with per-person jersey fields and track summaries."""
    gate_cfg = GatekeeperConfig(
        kit_hist_gate=config.kit_hist_gate,
        kit_pixel_gate=config.kit_pixel_gate,
        kit_pixel_tolerance=config.kit_pixel_tolerance,
    )
    gatekeeper = Gatekeeper(gate_cfg)
    last_read: dict[int, int] = {}
    track_agg: dict[int, _TrackAgg] = defaultdict(_TrackAgg)
    stats = {
        "geometry_rejects": 0,
        "kit_rejects": 0,
        "jersey_reads": 0,
        "labeled_detections": 0,
    }

    frames_data = sorted(payload.get("frames", []), key=lambda f: f["frame"])
    if max_frames is not None:
        frames_data = frames_data[:max_frames]

    total = len(frames_data)
    for i, frame_data in enumerate(frames_data):
        frame_idx = int(frame_data["frame"])
        frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
        frame_img = cv2.imread(str(frame_path))
        if frame_img is None:
            raise FileNotFoundError(f"Missing frame image: {frame_path}")

        enriched_people: list[dict[str, Any]] = []
        for person in frame_data.get("people", []):
            track_id = int(person["id"])
            bbox = normalize_bbox(person["bbox"], frame_img.shape)
            entry: dict[str, Any] = {
                "id": track_id,
                "bbox": [round(v, 2) for v in bbox],
                "conf": person.get("conf", 0.0),
            }

            if not is_valid_person_bbox(bbox, frame_img.shape):
                entry["gate"] = "geometry_reject"
                stats["geometry_rejects"] += 1
                enriched_people.append(entry)
                continue

            if config.use_kit_gate:
                gate = gatekeeper.evaluate(
                    frame_img,
                    bbox,
                    config.primary_color,
                    config.secondary_color,
                )
                read_bbox = gate.bbox
                entry["gate_passed"] = gate.passed
                if not gate.passed:
                    entry["gate"] = "kit_reject"
                    stats["kit_rejects"] += 1
                    enriched_people.append(entry)
                    continue
            else:
                read_bbox = bbox
                entry["gate_passed"] = True

            should_read = frame_idx - last_read.get(track_id, -9999) >= config.jersey_every
            if should_read:
                last_read[track_id] = frame_idx
                num, conf, source = reader.read_number_detailed(frame_img, read_bbox)
                stats["jersey_reads"] += 1
                entry["jersey_number"] = num
                entry["jersey_conf"] = round(conf, 4)
                entry["jersey_source"] = source
                track_agg[track_id].add(frame_idx, num, conf, source)
                if num is not None:
                    stats["labeled_detections"] += 1
            enriched_people.append(entry)

        frame_data["people"] = enriched_people
        if on_progress is not None:
            on_progress(i + 1, total)

    tracks_out: dict[str, dict[str, Any]] = {}
    for tid, agg in track_agg.items():
        num, mean_conf, src = agg.best_number()
        tracks_out[str(tid)] = {
            "jersey_number": num,
            "jersey_conf_mean": round(mean_conf, 4),
            "jersey_source": src,
            "read_count": len(agg.reads),
        }

    payload["tracks"] = tracks_out
    payload["label_stats"] = stats
    return payload
