"""Tests for frame_overlay drawing helpers."""

from __future__ import annotations

import numpy as np

from backend.app.pipeline.frame_overlay import (
    TrackDraw,
    draw_overlay_badminton,
    draw_tracks,
)


def test_draw_tracks_returns_copy_and_mutates_nothing() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[0, 0] = (10, 20, 30)
    tracks = [
        TrackDraw(track_id=1, bbox=[40.0, 20.0, 80.0, 100.0], label="10"),
        TrackDraw(track_id=2, bbox=[90.0, 30.0, 120.0, 110.0]),
    ]
    out = draw_tracks(
        frame,
        tracks,
        locked_track_id=1,
        ball_bbox=[10.0, 10.0, 20.0, 20.0],
    )
    assert out is not frame
    assert frame[0, 0].tolist() == [10, 20, 30]
    assert out.shape == frame.shape
    assert int(out.sum()) > 0


def test_draw_tracks_empty_returns_copy() -> None:
    frame = np.full((64, 64, 3), 128, dtype=np.uint8)
    out = draw_tracks(frame, [], locked_track_id=None)
    assert out is not frame
    np.testing.assert_array_equal(out, frame)


def test_draw_overlay_badminton_locked_label() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    tracks = [
        TrackDraw(track_id=1, bbox=[40.0, 20.0, 80.0, 100.0]),
        TrackDraw(track_id=2, bbox=[90.0, 30.0, 120.0, 110.0]),
    ]
    out = draw_overlay_badminton(
        frame,
        tracks,
        locked_track_id=1,
        court_side_label="NEAR",
        shuttle_px=None,
        rally_state="IDLE",
    )
    assert out is not frame
    assert int(out.sum()) > 0


def test_draw_overlay_badminton_shuttle_optional() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    out = draw_overlay_badminton(
        frame,
        [],
        locked_track_id=None,
        court_side_label="FAR",
        shuttle_px=(80.0, 60.0),
        rally_state="LIVE",
    )
    assert out is not frame
    assert int(out.sum()) > 0


def test_draw_overlay_badminton_rally_end_label() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    tracks = [
        TrackDraw(track_id=1, bbox=[40.0, 20.0, 80.0, 100.0]),
    ]
    out = draw_overlay_badminton(
        frame,
        tracks,
        locked_track_id=1,
        court_side_label="NEAR",
        shuttle_px=None,
        rally_state="END",
    )
    assert out is not frame
    assert int(out.sum()) > 0
