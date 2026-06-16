"""Tests for replay-suspect rally filtering helpers."""

from backend.app.pipeline.badminton.rally_tracker import RallyEvent
from backend.app.pipeline.run import (
    _filter_rallies_replay_suspect,
    _rally_overlaps_replay_suspect,
)


def test_rally_overlaps_suspect_when_end_near_cut():
    ev = RallyEvent(0, 100, "near", "near", False)
    suspect = {95}
    assert _rally_overlaps_replay_suspect(ev, suspect, window=10)


def test_rally_overlaps_suspect_when_cut_inside_rally():
    ev = RallyEvent(50, 150, "near", "near", False)
    suspect = {100}
    assert _rally_overlaps_replay_suspect(ev, suspect, window=5)


def test_filter_removes_rally_overlapping_suspect_window():
    events = [
        RallyEvent(0, 50, "near", "near", False),
        RallyEvent(500, 550, "far", "far", False),
    ]
    cut_frames = [498, 499, 500, 501, 502]
    filtered = _filter_rallies_replay_suspect(events, cut_frames)
    assert len(filtered) == 1
    assert filtered[0].start_frame == 0
