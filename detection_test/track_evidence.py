from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class JerseyRead:
    frame: int
    number: int
    conf: float
    source: str


@dataclass
class AccumulatorConfig:
    min_reads: int = 3
    window_frames: int = 90
    min_spacing: int = 5
    high_conf_classifier: float = 0.75
    high_conf_ocr: float = 0.55
    target_number_min_conf: float = 0.25


@dataclass
class TrackEvidenceBuffer:
    """Stage 2: collect independent high-confidence jersey reads before confirmation."""

    config: AccumulatorConfig
    reads: deque[JerseyRead] = field(default_factory=deque)
    confirmed: bool = False
    confirmed_at_frame: int | None = None

    def _min_conf_for_source(self, source: str) -> float:
        if source == "classifier":
            return self.config.high_conf_classifier
        if source == "classifier_target":
            return self.config.target_number_min_conf
        if source == "ocr":
            return self.config.high_conf_ocr
        return 1.0

    def _prune(self, frame_idx: int) -> None:
        cutoff = frame_idx - self.config.window_frames
        while self.reads and self.reads[0].frame < cutoff:
            self.reads.popleft()

    def _independent_reads(self, target_number: int) -> list[JerseyRead]:
        matching = [r for r in self.reads if r.number == target_number]
        if not matching:
            return []
        independent: list[JerseyRead] = []
        last_frame = -9999
        for read in matching:
            if read.frame - last_frame >= self.config.min_spacing:
                independent.append(read)
                last_frame = read.frame
        return independent

    def try_add_read(
        self,
        frame_idx: int,
        number: int | None,
        conf: float,
        source: str,
        target_number: int,
    ) -> bool:
        """Append a high-confidence read if it passes spacing rules. Returns True if added."""
        if number != target_number:
            self._prune(frame_idx)
            return False

        min_conf = self._min_conf_for_source(source)
        if conf < min_conf:
            self._prune(frame_idx)
            return False

        if self.reads and frame_idx - self.reads[-1].frame < self.config.min_spacing:
            self._prune(frame_idx)
            return False

        self.reads.append(JerseyRead(frame=frame_idx, number=number, conf=conf, source=source))
        self._prune(frame_idx)
        self._try_confirm(frame_idx, target_number)
        return True

    def _try_confirm(self, frame_idx: int, target_number: int) -> None:
        if self.confirmed:
            return
        independent = self._independent_reads(target_number)
        if len(independent) >= self.config.min_reads:
            self.confirmed = True
            self.confirmed_at_frame = frame_idx

    def independent_read_count(self, target_number: int) -> int:
        return len(self._independent_reads(target_number))

    def mean_confidence(self, target_number: int) -> float:
        reads = self._independent_reads(target_number)
        if not reads:
            return 0.0
        return sum(r.conf for r in reads) / len(reads)
