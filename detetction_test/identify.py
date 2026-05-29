from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from color_score import jersey_color_score
from face_match import FaceMatcher, NullFaceMatcher
from gatekeeper import Gatekeeper, GatekeeperConfig, GatekeeperResult
from jersey_number import JerseyProfileReader
from track_evidence import AccumulatorConfig, TrackEvidenceBuffer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class State(str, Enum):
    UNLOCKED = "unlocked"
    LOCKED = "locked"
    LOST = "lost"


@dataclass
class IdentifyConfig:
    face_weight: float = 2.0
    num_weight: float = 1.5
    name_weight: float = 0.8
    color_weight: float = 0.2
    lock_threshold: float = 1.8
    margin: float = 0.5
    initial_frames: int = 30
    initial_face_score: float = 3.0
    relock_window: int = 5
    relock_cooldown: int = 10
    relock_max: int = 300
    ocr_every: int = 5
    ocr_votes: float = 1.5
    ocr_reads: int = 3
    name_min_sim: float = 0.6
    color_gate: float = 0.1
    color_tolerance: int = 10
    absence_frames: int = 30
    number_min_conf: float = 0.75
    zero_face_unlock: int = 30
    relock_face_score: float = 2.0
    handoff_face_min: float = 0.45
    handoff_margin: float = 0.25
    handoff_max_frames: int = 60
    kit_hist_gate: float = 0.35
    min_evidence_reads: int = 3
    evidence_window_frames: int = 90
    min_read_spacing: int = 5
    high_conf_classifier: float = 0.75
    high_conf_ocr: float = 0.55
    target_number_min_conf: float = 0.25
    kit_pixel_gate: float = 0.12
    kit_pixel_tolerance: int = 40
    skip_kit_gate: bool = False
    require_confirmed_for_lock: bool = True


@dataclass
class TargetProfile:
    name: str
    number: int
    primary_color: str
    secondary_color: str
    confirmed_number: int | None = None
    confirmed_name: str | None = None


@dataclass
class LockedSegment:
    start_frame: int
    end_frame: int | None = None
    track_id: int = 0


@dataclass
class IdentifyRow:
    frame: int
    t: float
    track: int
    bbox: list[float]
    score: float
    method: str


def _name_letters(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"[^A-Za-z]", "", text).upper()


def name_similarity(observed: str | None, target: str) -> float:
    """Fuzzy match between OCR name and user-provided name."""
    a = _name_letters(observed)
    b = _name_letters(target)
    if not a or not b:
        return 0.0
    if a == b or b in a or a in b:
        return 1.0
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    best = 0
    for i in range(len(long) - len(short) + 1):
        match = sum(1 for j in range(len(short)) if long[i + j] == short[j])
        if match > best:
            best = match
    return best / len(short)


def _bbox_area(bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _method_string(
    face: float,
    number_conf: float,
    name_sim: float,
    color: float,
) -> str:
    parts: list[str] = []
    if face > 0:
        parts.append("face")
    if number_conf > 0:
        parts.append("number")
    if name_sim > 0:
        parts.append("name")
    if color > 0:
        parts.append("color")
    return "+".join(parts) if parts else "none"


@dataclass
class IdentifyEngine:
    config: IdentifyConfig
    profile: TargetProfile
    face_matcher: FaceMatcher | NullFaceMatcher
    ocr: JerseyProfileReader

    state: State = State.UNLOCKED
    locked_track_id: int | None = None
    locked_segment: LockedSegment | None = None
    locked_segments: list[LockedSegment] = field(default_factory=list)
    rows: list[IdentifyRow] = field(default_factory=list)

    cumulative_face: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    track_evidence: dict[int, TrackEvidenceBuffer] = field(default_factory=dict)
    number_reads: dict[str, tuple[float, int]] = field(default_factory=dict)
    name_match_count: int = 0
    last_seen_locked_frame: int | None = None
    last_ocr_frame: int = -999
    lost_at_frame: int | None = None
    lost_scan_frames: int = 0
    relock_face_cumulative: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    zero_face_streak: int = 0
    recent_bbox_areas: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    gave_up: bool = False
    _frames_processed: int = 0
    gatekeeper_rejects: int = 0
    gatekeeper_passes: int = 0
    _last_jersey_read_frame: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._gatekeeper = Gatekeeper(
            GatekeeperConfig(
                kit_hist_gate=self.config.kit_hist_gate,
                kit_pixel_gate=self.config.kit_pixel_gate,
                kit_pixel_tolerance=self.config.kit_pixel_tolerance,
                skip_kit_gate=self.config.skip_kit_gate,
            )
        )
        self._accumulator_config = AccumulatorConfig(
            min_reads=self.config.min_evidence_reads,
            window_frames=self.config.evidence_window_frames,
            min_spacing=self.config.min_read_spacing,
            high_conf_classifier=self.config.high_conf_classifier,
            high_conf_ocr=self.config.high_conf_ocr,
            target_number_min_conf=self.config.target_number_min_conf,
        )

    def _evidence_for(self, track_id: int) -> TrackEvidenceBuffer:
        buf = self.track_evidence.get(track_id)
        if buf is None:
            buf = TrackEvidenceBuffer(config=self._accumulator_config)
            self.track_evidence[track_id] = buf
        return buf

    def _weighted_score(
        self,
        face: float,
        number_conf: float,
        name_sim: float,
        color: float,
    ) -> float:
        c = self.config
        return (
            c.face_weight * face
            + c.num_weight * number_conf
            + c.name_weight * name_sim
            + c.color_weight * color
        )

    def _run_gatekeeper(
        self,
        frame_img: NDArray,
        bbox: list[float],
    ) -> GatekeeperResult:
        result = self._gatekeeper.evaluate(
            frame_img,
            bbox,
            self.profile.primary_color,
            self.profile.secondary_color,
        )
        if result.passed:
            self.gatekeeper_passes += 1
        else:
            self.gatekeeper_rejects += 1
        return result

    def _jersey_read_interval(self) -> int:
        return max(self.config.min_read_spacing, self.config.ocr_every)

    def _should_read_jersey(self, track_id: int, frame_idx: int) -> bool:
        last = self._last_jersey_read_frame.get(track_id, -9999)
        if frame_idx - last < self._jersey_read_interval():
            return False
        self._last_jersey_read_frame[track_id] = frame_idx
        return True

    def _read_jersey(
        self,
        frame_img: NDArray,
        bbox: list[float],
        *,
        gate: GatekeeperResult | None = None,
        target_number: int | None = None,
    ) -> tuple[GatekeeperResult, int | None, float, str]:
        if gate is None:
            gate = self._run_gatekeeper(frame_img, bbox)
        if not gate.passed:
            return gate, None, 0.0, "gated"
        target = target_number if target_number is not None else self.profile.number
        num, conf, source = self.ocr.read_number_detailed(
            frame_img, gate.bbox, target_number=target
        )
        return gate, num, conf, source

    def _read_jersey_if_gated(
        self,
        frame_img: NDArray,
        bbox: list[float],
    ) -> tuple[GatekeeperResult, int | None, float, str]:
        return self._read_jersey(frame_img, bbox)

    def _accumulate_jersey_read(
        self,
        track_id: int,
        frame_idx: int,
        number: int | None,
        conf: float,
        source: str,
        *,
        target_number: int | None = None,
    ) -> TrackEvidenceBuffer:
        target = target_number if target_number is not None else self.profile.number
        buf = self._evidence_for(track_id)
        buf.try_add_read(frame_idx, number, conf, source, target)
        if buf.confirmed and self.profile.confirmed_number is None:
            self.profile.confirmed_number = target
        return buf

    def _score_signals_gated(
        self,
        frame_img: NDArray,
        bbox: list[float],
    ) -> tuple[GatekeeperResult, float, float]:
        """Cheap per-frame signals only — jersey OCR runs in _observe_ocr on ocr_every."""
        gate = self._run_gatekeeper(frame_img, bbox)
        face = self.face_matcher.score_bbox(frame_img, gate.bbox)
        color = jersey_color_score(
            frame_img,
            gate.bbox,
            self.profile.primary_color,
            self.profile.secondary_color,
            tolerance=self.config.color_tolerance,
        )
        return gate, face, color

    def _observe_ocr(
        self,
        frame_img: NDArray,
        gate: GatekeeperResult,
        frame_idx: int,
        track_id: int,
        *,
        for_relock: bool = False,
    ) -> tuple[float, float]:
        c = self.config
        if frame_idx - self.last_ocr_frame < c.ocr_every:
            return 0.0, 0.0
        self.last_ocr_frame = frame_idx

        if not gate.passed:
            return 0.0, 0.0

        target_number = (
            (self.profile.confirmed_number or self.profile.number)
            if for_relock
            else self.profile.number
        )
        num, num_conf, source = self.ocr.read_number_detailed(
            frame_img, gate.bbox, target_number=target_number
        )
        self._accumulate_jersey_read(
            track_id,
            frame_idx,
            num,
            num_conf,
            source,
            target_number=target_number if for_relock else None,
        )

        number_conf = 0.0
        if num == target_number and num_conf >= c.number_min_conf:
            number_conf = num_conf

        if num == self.profile.number and num_conf >= c.high_conf_classifier:
            key = str(num)
            prev_conf, prev_count = self.number_reads.get(key, (0.0, 0))
            self.number_reads[key] = (prev_conf + num_conf, prev_count + 1)

        name, _name_conf = self.ocr.read_name(frame_img, gate.bbox)
        name_sim = name_similarity(name, self.profile.name)
        if name_sim >= c.name_min_sim:
            self.name_match_count += 1
            if (
                self.profile.confirmed_name is None
                and self.name_match_count >= c.ocr_reads
            ):
                self.profile.confirmed_name = self.profile.name
        else:
            name_sim = 0.0
        return number_conf, name_sim

    def _lock_track(self, track_id: int, frame_idx: int, *, new_segment: bool = True) -> None:
        self.state = State.LOCKED
        self.locked_track_id = track_id
        if new_segment or self.locked_segment is None:
            self.locked_segment = LockedSegment(start_frame=frame_idx, track_id=track_id)
        else:
            self.locked_segment.track_id = track_id
        self.last_seen_locked_frame = frame_idx
        self.last_ocr_frame = -999
        if new_segment:
            self.number_reads.clear()
            self.name_match_count = 0
        self.relock_face_cumulative.clear()
        self.zero_face_streak = 0
        self.recent_bbox_areas.clear()

    def _try_handoff(
        self,
        frame_img: NDArray,
        people: list[dict[str, Any]],
        frame_idx: int,
    ) -> bool:
        c = self.config
        if not people:
            return False

        people_valid = self._filter_people_gated(frame_img, people)
        face_scores = self.face_matcher.score_frame(frame_img, people_valid)
        if not face_scores:
            return False

        best_id, best_face = max(face_scores.items(), key=lambda kv: kv[1])
        if best_face < c.handoff_face_min:
            return False

        sorted_faces = sorted(face_scores.values(), reverse=True)
        second = sorted_faces[1] if len(sorted_faces) > 1 else 0.0
        if best_face - second < c.handoff_margin:
            return False

        person = next(p for p in people_valid if int(p["id"]) == best_id)
        color = jersey_color_score(
            frame_img,
            person["bbox"],
            self.profile.primary_color,
            self.profile.secondary_color,
            tolerance=c.color_tolerance,
        )
        if color < c.color_gate and best_face < 0.6:
            return False

        self._lock_track(best_id, frame_idx, new_segment=False)
        return True

    def _should_unlock_ghost_track(self, bbox: list[float]) -> bool:
        c = self.config
        if self.zero_face_streak < c.zero_face_unlock:
            return False
        if not self.recent_bbox_areas:
            return True
        median_area = sorted(self.recent_bbox_areas)[len(self.recent_bbox_areas) // 2]
        return _bbox_area(bbox) < median_area * 0.25

    def _close_segment(self, end_frame: int) -> None:
        if self.locked_segment is not None:
            self.locked_segment.end_frame = end_frame
            self.locked_segments.append(self.locked_segment)
            self.locked_segment = None

    def _enter_lost(self, frame_idx: int) -> None:
        end = self.last_seen_locked_frame if self.last_seen_locked_frame is not None else frame_idx
        self._close_segment(end)
        self.state = State.LOST
        self.lost_at_frame = frame_idx
        self.lost_scan_frames = 0
        self.locked_track_id = None
        self.relock_face_cumulative.clear()
        self.zero_face_streak = 0

    def _confirmed_tracks(self) -> list[tuple[int, TrackEvidenceBuffer]]:
        return [
            (tid, buf)
            for tid, buf in self.track_evidence.items()
            if buf.confirmed
        ]

    def _pick_best_confirmed_track(self) -> int | None:
        confirmed = self._confirmed_tracks()
        if not confirmed:
            return None
        target = self.profile.number

        def rank(item: tuple[int, TrackEvidenceBuffer]) -> tuple[float, float, int]:
            tid, buf = item
            return (
                buf.independent_read_count(target),
                buf.mean_confidence(target),
                -tid,
            )

        best_id, _buf = max(confirmed, key=rank)
        return best_id

    def _try_initial_lock_face(self, frame_idx: int) -> None:
        c = self.config
        if not self.cumulative_face:
            return

        if c.require_confirmed_for_lock:
            confirmed_id = self._pick_best_confirmed_track()
            if confirmed_id is not None:
                self._lock_track(confirmed_id, frame_idx)
            return

        best_id, best_score = max(self.cumulative_face.items(), key=lambda kv: kv[1])
        if best_score >= c.initial_face_score:
            self._lock_track(best_id, frame_idx)
            return

        if frame_idx + 1 < c.initial_frames:
            return

        sorted_scores = sorted(self.cumulative_face.values(), reverse=True)
        if len(sorted_scores) >= 2:
            margin_ok = (sorted_scores[0] - sorted_scores[1]) >= c.margin
        else:
            margin_ok = sorted_scores[0] > 0
        if margin_ok and best_score > 0:
            self._lock_track(best_id, frame_idx)

    def _try_initial_lock_confirmed(self, frame_idx: int) -> None:
        if not self.config.require_confirmed_for_lock:
            return
        best_id = self._pick_best_confirmed_track()
        if best_id is not None:
            self._lock_track(best_id, frame_idx)

    def _try_relock(self, frame_idx: int, people_by_id: dict[int, dict[str, Any]]) -> None:
        c = self.config
        if c.face_weight > 0 and self.relock_face_cumulative:
            visible = {
                tid: score
                for tid, score in self.relock_face_cumulative.items()
                if tid in people_by_id
            }
            if visible:
                best_id, best_score = max(visible.items(), key=lambda kv: kv[1])
                if best_score >= c.relock_face_score:
                    sorted_scores = sorted(visible.values(), reverse=True)
                    second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
                    if best_score - second >= c.margin:
                        self._lock_track(best_id, frame_idx, new_segment=True)
                        return

        if c.require_confirmed_for_lock:
            best_id = self._pick_best_confirmed_track()
            if best_id is not None and best_id in people_by_id:
                self._lock_track(best_id, frame_idx, new_segment=True)

    def _emit_row(
        self,
        frame_idx: int,
        timestamp: float,
        track_id: int,
        bbox: list[float],
        face: float,
        number_conf: float,
        name_sim: float,
        color: float,
    ) -> None:
        score = self._weighted_score(face, number_conf, name_sim, color)
        self.rows.append(
            IdentifyRow(
                frame=frame_idx,
                t=round(timestamp, 4),
                track=track_id,
                bbox=[round(v, 2) for v in bbox],
                score=round(score, 4),
                method=_method_string(face, number_conf, name_sim, color),
            )
        )

    def _filter_people_gated(
        self,
        frame_img: NDArray,
        people: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for person in people:
            gate = self._run_gatekeeper(frame_img, person["bbox"])
            if not gate.passed:
                continue
            out.append({**person, "bbox": gate.bbox})
        return out

    def _seed_jersey_evidence(
        self,
        frame_img: NDArray,
        people: list[dict[str, Any]],
        frame_idx: int,
    ) -> None:
        """Accumulate jersey reads for gate-passed detections (throttled per track)."""
        for person in people:
            track_id = int(person["id"])
            if not self._should_read_jersey(track_id, frame_idx):
                continue
            num, conf, source = self.ocr.read_number_detailed(
                frame_img, person["bbox"], target_number=self.profile.number
            )
            self._accumulate_jersey_read(track_id, frame_idx, num, conf, source)

    def process_frame(
        self,
        frame_idx: int,
        timestamp: float,
        frame_img: NDArray,
        people: list[dict[str, Any]],
    ) -> None:
        c = self.config
        self._frames_processed += 1
        people_by_id = {int(p["id"]): p for p in people}

        if self.state == State.UNLOCKED:
            if c.face_weight > 0:
                people_valid = self._filter_people_gated(frame_img, people)
                face_scores = self.face_matcher.score_frame(frame_img, people_valid)
                for track_id, face_sim in face_scores.items():
                    self.cumulative_face[track_id] += face_sim
                self._seed_jersey_evidence(frame_img, people_valid, frame_idx)
                self._try_initial_lock_face(frame_idx)
            else:
                for person in people:
                    track_id = int(person["id"])
                    if not self._should_read_jersey(track_id, frame_idx):
                        continue
                    gate, num, conf, source = self._read_jersey_if_gated(
                        frame_img, person["bbox"]
                    )
                    if gate.passed:
                        self._accumulate_jersey_read(
                            track_id, frame_idx, num, conf, source
                        )
                self._try_initial_lock_confirmed(frame_idx)
            return

        if self.state == State.LOCKED:
            assert self.locked_track_id is not None
            person = people_by_id.get(self.locked_track_id)
            if person is None:
                if self._try_handoff(frame_img, people, frame_idx):
                    person = people_by_id.get(self.locked_track_id)
                elif (
                    self.last_seen_locked_frame is not None
                    and frame_idx - self.last_seen_locked_frame > c.handoff_max_frames
                ):
                    self._enter_lost(frame_idx)
                    return
                elif person is None:
                    return

            self.last_seen_locked_frame = frame_idx
            bbox = person["bbox"]
            gate, face, color = self._score_signals_gated(frame_img, bbox)
            if not gate.geometry_ok:
                self._enter_lost(frame_idx)
                return
            number_conf, name_sim = self._observe_ocr(
                frame_img,
                gate,
                frame_idx,
                self.locked_track_id,
            )
            if face <= 0.0:
                self.zero_face_streak += 1
            else:
                self.zero_face_streak = 0
                self.recent_bbox_areas.append(_bbox_area(gate.bbox))
            if self._should_unlock_ghost_track(gate.bbox):
                self._enter_lost(frame_idx)
                return
            self._emit_row(
                frame_idx, timestamp, self.locked_track_id, gate.bbox,
                face, number_conf, name_sim, color,
            )
            return

        if self.state == State.LOST:
            if self.lost_at_frame is None:
                return
            elapsed = frame_idx - self.lost_at_frame
            if elapsed <= c.relock_cooldown:
                return
            if elapsed > c.relock_max:
                self.gave_up = True
                return

            self.lost_scan_frames += 1
            target_number = self.profile.confirmed_number or self.profile.number

            for person in people:
                track_id = int(person["id"])
                if not self._should_read_jersey(track_id, frame_idx):
                    continue
                gate, num, conf, source = self._read_jersey_if_gated(
                    frame_img, person["bbox"]
                )
                if not gate.passed:
                    continue

                self._accumulate_jersey_read(
                    track_id,
                    frame_idx,
                    num,
                    conf,
                    source,
                    target_number=target_number,
                )

                if c.face_weight > 0:
                    face = self.face_matcher.score_bbox(frame_img, gate.bbox)
                    if face > 0.0:
                        self.relock_face_cumulative[track_id] += face

            self._try_relock(frame_idx, people_by_id)

    def finalize(self) -> None:
        if self.state == State.LOCKED and self.locked_segment is not None:
            end = self.last_seen_locked_frame if self.last_seen_locked_frame is not None else self.locked_segment.start_frame
            self._close_segment(end)

    def summary(self, total_frames: int) -> dict[str, Any]:
        locked_frames = len(self.rows)
        confirmed_count = sum(1 for b in self.track_evidence.values() if b.confirmed)
        return {
            "locked_frames": locked_frames,
            "unlocked_frames": max(0, total_frames - locked_frames),
            "lock_count": len(self.locked_segments),
            "gave_up": self.gave_up,
            "confirmed_tracks": confirmed_count,
            "gatekeeper_passes": self.gatekeeper_passes,
            "gatekeeper_rejects": self.gatekeeper_rejects,
        }

    def evidence_diagnostics(self) -> dict[str, Any]:
        target = self.profile.number
        per_track: dict[str, dict[str, Any]] = {}
        for tid, buf in sorted(self.track_evidence.items()):
            if not buf.reads:
                continue
            per_track[str(tid)] = {
                "confirmed": buf.confirmed,
                "independent_reads": buf.independent_read_count(target),
                "mean_conf": round(buf.mean_confidence(target), 4),
                "total_reads": len(buf.reads),
            }
        return per_track
