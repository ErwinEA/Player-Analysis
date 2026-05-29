"""Tests for run_pipeline helpers (frame cap, heatmap row selection)."""

from __future__ import annotations

import os

from backend.app.pipeline.run import (
    DEFAULT_MAX_FRAMES,
    MAX_FRAMES_ENV_CAP,
    _ball_events_enabled,
    _max_frames,
    _rows_for_analytics,
    _rows_for_heatmap,
)
from backend.app.schemas import Row


def _row(track: int, frame: int) -> Row:
    return Row(
        t=float(frame),
        frame=frame,
        track=track,
        player=None,
        x=0.5,
        y=0.5,
        bbox=[0.0, 0.0, 10.0, 10.0],
    )


def test_max_frames_default(monkeypatch):
    monkeypatch.delenv("MAX_FRAMES", raising=False)
    assert _max_frames() == DEFAULT_MAX_FRAMES


def test_max_frames_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("MAX_FRAMES", "not-a-number")
    assert _max_frames() == DEFAULT_MAX_FRAMES


def test_max_frames_zero_falls_back(monkeypatch):
    monkeypatch.setenv("MAX_FRAMES", "0")
    assert _max_frames() == DEFAULT_MAX_FRAMES


def test_max_frames_capped(monkeypatch):
    monkeypatch.setenv("MAX_FRAMES", "99999")
    assert _max_frames() == MAX_FRAMES_ENV_CAP


def test_rows_for_heatmap_locked_target():
    rows = [_row(1, 0), _row(2, 1), _row(1, 2)]
    out, source = _rows_for_heatmap(rows, 1)
    assert source == "locked_target"
    assert all(r.track == 1 for r in out)
    assert len(out) == 2


def test_rows_for_heatmap_by_jersey_includes_reid_fallback_track():
    rows = [
        Row(
            t=0.0,
            frame=0,
            track=99,
            player=10,
            x=0.5,
            y=0.5,
            bbox=[0.0, 0.0, 10.0, 10.0],
            segment_source="reid_fallback",
        ),
        _row(1, 1),
    ]
    out, source = _rows_for_heatmap(rows, 1, target_jersey=10)
    assert source == "locked_target"
    assert len(out) == 1
    assert out[0].track == 99


def test_rows_for_heatmap_fallback_track():
    rows = [_row(1, 0), _row(2, 1), _row(2, 2), _row(2, 3)]
    out, source = _rows_for_heatmap(rows, None)
    assert source == "fallback_track"
    assert all(r.track == 2 for r in out)


def test_rows_for_analytics_from_lock_excludes_gap_fill():
    rows = [
        _row(1, 5),
        _row(1, 10),
        Row(
            t=11.0,
            frame=11,
            track=1,
            player=10,
            x=0.5,
            y=0.5,
            bbox=[0.0, 0.0, 10.0, 10.0],
            segment_source="gap_fill",
            mask_fallback=True,
        ),
        _row(1, 12),
    ]
    out = _rows_for_analytics(rows, locked_at_frame=10)
    assert len(out) == 2
    assert all(r.frame >= 10 for r in out)
    assert all(r.segment_source != "gap_fill" for r in out)


def test_gap_fill_rows_dropped_for_analytics_only():
    """Gap-fill duplicates one spot on the heatmap but add no path length."""
    rows = [
        Row(t=0.0, frame=0, track=1, player=10, x=0.5, y=0.5, bbox=[0, 0, 10, 10]),
        Row(t=1.0, frame=30, track=1, player=10, x=0.5, y=0.5, bbox=[0, 0, 10, 18]),
    ]
    gap_spam = [
        Row(
            t=1.0 + i * 0.01,
            frame=30 + i,
            track=1,
            player=10,
            x=0.5,
            y=0.5,
            bbox=[0, 0, 10, 18],
            segment_source="gap_fill",
            mask_fallback=True,
        )
        for i in range(50)
    ]
    with_gaps = rows + gap_spam
    analytics = _rows_for_analytics(with_gaps, locked_at_frame=None)
    assert len(analytics) == 2
    assert len(with_gaps) == 52


def test_ball_events_enabled_kill_switch(monkeypatch):
    monkeypatch.setenv("ENABLE_BALL_EVENTS", "0")
    assert _ball_events_enabled() is False
    monkeypatch.setenv("ENABLE_BALL_EVENTS", "1")
    assert _ball_events_enabled() is True
