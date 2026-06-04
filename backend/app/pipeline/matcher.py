from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


def _color_lock_threshold() -> float:
    return float(os.environ.get("COLOR_LOCK_THRESHOLD", "0.15"))


def _color_min_floor() -> float:
    return float(os.environ.get("COLOR_MATCH_MIN", "0.05"))


def _name_min_ratio() -> float:
    return float(os.environ.get("NAME_MIN_RATIO", "0.6"))


def _name_min_conf() -> float:
    return float(os.environ.get("NAME_MIN_CONF", "0.35"))


def _number_min_conf() -> float:
    return float(os.environ.get("NUMBER_MIN_CONF", "0.40"))


def _number_lock_votes() -> float:
    """Cumulative number-OCR confidence at which we lock by number alone."""
    return float(os.environ.get("NUMBER_LOCK_VOTES", "1.0"))


def _reid_lock_threshold() -> float:
    return float(os.environ.get("REID_LOCK_THRESHOLD", "0.65"))


def _reid_min_samples() -> int:
    return max(1, int(os.environ.get("REID_MIN_SAMPLES", "2")))


def _reid_prototype_min_conf() -> float:
    return float(os.environ.get("REID_PROTOTYPE_MIN_CONF", "0.40"))


def _name_letters(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"[^A-Za-z]", "", text).upper()


def _name_similarity(observed: str | None, target: str) -> float:
    """Cheap fuzzy match: contiguous overlap ratio between cleaned strings."""
    a = _name_letters(observed)
    b = _name_letters(target)
    if not a or not b:
        return 0.0
    if a == b or b in a or a in b:
        return 1.0
    # Sliding-window character overlap
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    best = 0
    for i in range(len(long) - len(short) + 1):
        match = sum(1 for j in range(len(short)) if long[i + j] == short[j])
        if match > best:
            best = match
    return best / len(short)


@dataclass
class TargetLock:
    track_id: int
    jersey: int
    method: str  # "number+name" | "number" | "reid" | "color"
    confidence: float


class TrackIdentityMatcher:
    """Three-stage identifier with permanent track lock once a player is identified."""

    def __init__(self, target_jersey: int, target_name: str) -> None:
        self.target_jersey = target_jersey
        self.target_name = _name_letters(target_name)

        self.number_votes: dict[int, float] = defaultdict(float)
        self.name_votes: dict[int, float] = defaultdict(float)
        self.best_name_for_track: dict[int, tuple[str, float]] = {}
        self.color_scores: dict[int, list[float]] = defaultdict(list)
        self.has_number_match: dict[int, bool] = defaultdict(bool)
        self.reid_embeddings: dict[int, list[np.ndarray]] = defaultdict(list)
        self.reid_prototype: np.ndarray | None = None

        self.lock: TargetLock | None = None

    @property
    def is_locked(self) -> bool:
        return self.lock is not None

    def release_lock(self) -> None:
        """Drop the current lock so the target can be re-acquired on a new shot.

        Keeps the ReID prototype and vote history by default for fast re-lock; set
        SCENE_CUT_CLEAR_VOTES=1 to also clear accumulated evidence when stale votes
        cause repeated wrong re-locks across cuts.
        """
        self.lock = None
        if os.environ.get("SCENE_CUT_CLEAR_VOTES", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            self.number_votes.clear()
            self.name_votes.clear()
            self.best_name_for_track.clear()
            self.color_scores.clear()
            self.has_number_match.clear()
            self.reid_embeddings.clear()

    def observe_number(
        self,
        track_id: int,
        number: int | None,
        conf: float,
        *,
        embedding: np.ndarray | None = None,
    ) -> None:
        if number is None or conf < _number_min_conf():
            return
        if number == self.target_jersey:
            self.number_votes[track_id] += conf
            self.has_number_match[track_id] = True
            if (
                self.reid_prototype is None
                and embedding is not None
                and conf >= _reid_prototype_min_conf()
            ):
                vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    self.reid_prototype = vec / norm

    def observe_reid(self, track_id: int, embedding: np.ndarray) -> None:
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            return
        norm = np.linalg.norm(vec)
        if norm <= 0:
            return
        self.reid_embeddings[track_id].append(vec / norm)

    def observe_name(self, track_id: int, name: str | None, conf: float) -> None:
        if not name or conf < _name_min_conf() or not self.target_name:
            return
        similarity = _name_similarity(name, self.target_name)
        if similarity < _name_min_ratio():
            return
        score = similarity * conf
        self.name_votes[track_id] += score
        prev = self.best_name_for_track.get(track_id)
        if prev is None or score > prev[1]:
            self.best_name_for_track[track_id] = (name, score)

    def observe_color(self, track_id: int, color_score: float) -> None:
        if color_score > 0:
            self.color_scores[track_id].append(color_score)

    def _try_lock_number_name(self) -> TargetLock | None:
        best_id: int | None = None
        best_score = 0.0
        for track_id, num_score in self.number_votes.items():
            name_score = self.name_votes.get(track_id, 0.0)
            if name_score <= 0:
                continue
            combined = num_score + name_score
            if combined > best_score:
                best_id = track_id
                best_score = combined
        if best_id is None:
            return None
        return TargetLock(
            track_id=best_id,
            jersey=self.target_jersey,
            method="number+name",
            confidence=round(best_score, 3),
        )

    def _try_lock_reid(self) -> TargetLock | None:
        if self.reid_prototype is None:
            return None
        from backend.app.pipeline.reid import cosine_similarity, mean_embedding

        threshold = _reid_lock_threshold()
        min_samples = _reid_min_samples()
        best_id: int | None = None
        best_sim = 0.0
        for track_id, embs in self.reid_embeddings.items():
            if not self.has_number_match.get(track_id):
                continue
            if len(embs) < min_samples:
                continue
            sim = cosine_similarity(mean_embedding(embs), self.reid_prototype)
            if sim > best_sim:
                best_id = track_id
                best_sim = sim
        if best_id is None or best_sim < threshold:
            return None
        return TargetLock(
            track_id=best_id,
            jersey=self.target_jersey,
            method="reid",
            confidence=round(best_sim, 3),
        )

    def _try_lock_number(self) -> TargetLock | None:
        threshold = _number_lock_votes()
        best_id: int | None = None
        best_score = 0.0
        for track_id, num_score in self.number_votes.items():
            if num_score >= threshold and num_score > best_score:
                best_id = track_id
                best_score = num_score
        if best_id is None:
            return None
        return TargetLock(
            track_id=best_id,
            jersey=self.target_jersey,
            method="number",
            confidence=round(best_score, 3),
        )

    def _try_lock_color(self) -> TargetLock | None:
        if not self.color_scores:
            return None
        threshold = _color_lock_threshold()
        floor = _color_min_floor()
        best_id: int | None = None
        best_avg = 0.0
        for track_id, scores in self.color_scores.items():
            if len(scores) < 5:
                continue
            avg = sum(scores) / len(scores)
            if avg > best_avg:
                best_id = track_id
                best_avg = avg
        if best_id is None or best_avg < max(threshold, floor):
            return None
        return TargetLock(
            track_id=best_id,
            jersey=self.target_jersey,
            method="color",
            confidence=round(best_avg, 3),
        )

    def _strongest_available_lock(self) -> TargetLock | None:
        """Best lock by evidence tier (ReID before weak color)."""
        lock = self._try_lock_number_name()
        if lock is None:
            lock = self._try_lock_number()
        if lock is None:
            lock = self._try_lock_reid()
        if lock is None:
            lock = self._try_lock_color()
        return lock

    def try_acquire_lock(self) -> TargetLock | None:
        """Acquire permanent lock: number+name → number → ReID → color."""
        if self.lock is not None:
            return self.lock
        lock = self._strongest_available_lock()
        if lock is not None:
            self.lock = lock
        return self.lock

    def maybe_upgrade_lock(self, track_id: int) -> TargetLock | None:
        """Promote a weak color lock when stronger evidence arrives on the same track."""
        if self.lock is None:
            return self.try_acquire_lock()
        if self.lock.method not in ("color",):
            return self.lock
        if self.lock.track_id != track_id:
            return self.lock
        for factory in (
            self._try_lock_number_name,
            self._try_lock_number,
            self._try_lock_reid,
        ):
            candidate = factory()
            if candidate is not None and candidate.track_id == track_id:
                self.lock = candidate
                return self.lock
        return self.lock

    def finalize(self) -> TargetLock | None:
        """End-of-video lock: ReID before color if still unlocked."""
        if self.lock is not None:
            return self.lock
        lock = self._try_lock_reid()
        if lock is None:
            lock = self._try_lock_color()
        if lock is not None:
            self.lock = lock
        return self.lock

    def soft_identify_track(
        self,
        *,
        prefer_track_id: int | None = None,
        reid_threshold: float = 0.5,
    ) -> tuple[int, str, float] | None:
        """Best pre-lock track for ball events: (track_id, id_type, score).

        Priority: continuity (prefer_track_id) → ReID → accumulated number OCR.
        Voting-only tracks without a number match are excluded.
        """
        if prefer_track_id is not None and self.has_number_match.get(prefer_track_id):
            return (
                prefer_track_id,
                "track",
                float(self.number_votes.get(prefer_track_id, 0.0)),
            )

        if self.reid_prototype is not None:
            from backend.app.pipeline.reid import cosine_similarity, mean_embedding

            best_id: int | None = None
            best_sim = 0.0
            for track_id, embs in self.reid_embeddings.items():
                if not self.has_number_match.get(track_id):
                    continue
                if not embs:
                    continue
                sim = cosine_similarity(mean_embedding(embs), self.reid_prototype)
                if sim > best_sim:
                    best_id = track_id
                    best_sim = sim
            if best_id is not None and best_sim >= reid_threshold:
                return best_id, "reid", best_sim

        best_id = None
        best_score = 0.0
        for track_id, num_score in self.number_votes.items():
            if not self.has_number_match.get(track_id):
                continue
            if num_score > best_score:
                best_id = track_id
                best_score = num_score
        if best_id is not None and best_score >= _number_min_conf():
            return best_id, "track", best_score

        return None
