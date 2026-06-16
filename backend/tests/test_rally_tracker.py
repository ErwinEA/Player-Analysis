"""Rally state machine transitions and win attribution."""

from backend.app.pipeline.badminton.rally_tracker import RallyTracker


def _tracker(**kwargs) -> RallyTracker:
    defaults = dict(
        net_y_px=360.0,
        fps=30.0,
        live_frames=2,
        gap_frames=5,
        post_land_cooldown_frames=0,
        merge_ms=0.0,
    )
    defaults.update(kwargs)
    return RallyTracker(**defaults)


def test_idle_to_live_to_end_transitions():
    tr = _tracker(land_stable_frames=3)
    assert tr.update(0, (100.0, 400.0)) == "IDLE"
    assert tr.update(1, (105.0, 410.0)) == "LIVE"
    assert tr.update(2, (110.0, 420.0)) == "LIVE"
    # Shuttle visible but stationary on court after flight => landing ends rally.
    assert tr.update(3, (110.0, 420.0)) == "LIVE"
    assert tr.update(4, (110.0, 420.0)) == "LIVE"
    assert tr.update(5, (110.0, 420.0)) == "END"
    assert tr.update(6, None) == "IDLE"
    assert len(tr.events) == 1
    ev = tr.events[0]
    assert ev.start_frame == 0
    assert ev.end_frame == 5


def test_gap_after_flight_ends_rally():
    tr = _tracker(land_stable_frames=10)
    tr.update(0, (100.0, 400.0))
    tr.update(1, (100.0, 410.0))
    for i in range(2, 6):
        tr.update(i, (100.0, 420.0))
    for i in range(6, 11):
        tr.update(i, None)
    assert len(tr.events) == 1
    assert tr.events[0].end_frame == 5


def test_win_attribution_landing_side_loses():
    # Shuttle last seen below the net line (near side) => far side wins.
    tr = _tracker()
    tr.update(0, (100.0, 500.0))
    tr.update(1, (100.0, 510.0))
    for i in range(2, 8):
        tr.update(i, None)
    assert len(tr.events) == 1
    assert tr.events[0].winner_side == "far"

    # Shuttle last seen above the net line (far side) => near side wins.
    tr2 = _tracker()
    tr2.update(0, (100.0, 100.0))
    tr2.update(1, (100.0, 110.0))
    for i in range(2, 8):
        tr2.update(i, None)
    assert tr2.events[0].winner_side == "near"


def test_locked_player_touch_tracked():
    tr = _tracker(touch_radius_px=50.0, touch_recent_frames=30)
    tr.update(0, (100.0, 500.0), locked_foot_px=(110.0, 510.0))
    tr.update(1, (100.0, 510.0), locked_foot_px=(110.0, 510.0))
    for i in range(2, 8):
        tr.update(i, None)
    assert tr.events[0].locked_player_touched_last is True

    tr2 = _tracker(touch_radius_px=50.0)
    tr2.update(0, (100.0, 500.0), locked_foot_px=(800.0, 100.0))
    tr2.update(1, (100.0, 510.0), locked_foot_px=(800.0, 100.0))
    for i in range(2, 8):
        tr2.update(i, None)
    assert tr2.events[0].locked_player_touched_last is False


def test_finalize_closes_live_rally():
    tr = _tracker()
    tr.update(0, (100.0, 400.0))
    tr.update(1, (100.0, 405.0))
    assert tr.state == "LIVE"
    tr.finalize(2)
    assert tr.state == "END"
    assert len(tr.events) == 1


def test_finalize_before_reset_preserves_live_rally():
    tr = _tracker()
    tr.update(0, (100.0, 400.0))
    tr.update(1, (100.0, 405.0))
    assert tr.state == "LIVE"
    tr.finalize(2)
    assert len(tr.events) == 1
    assert tr.events[0].end_frame == 1


def test_serving_side_alternates_each_rally():
    tr = _tracker()
    tr.update(0, (100.0, 400.0))
    tr.update(1, (100.0, 405.0))
    for i in range(2, 8):
        tr.update(i, None)
    assert len(tr.events) == 1
    assert tr.events[0].serving_side == "near"

    tr.update(8, (100.0, 400.0))
    tr.update(9, (100.0, 405.0))
    for i in range(10, 16):
        tr.update(i, None)
    assert len(tr.events) == 2
    assert tr.events[1].serving_side == "far"


def test_motion_latch_survives_short_gap():
    """One missed frame should not reset serve-motion latch before rally opens."""
    tr = _tracker(live_frames=2, start_gap_max=2, serve_speed_px=5.0)
    tr.update(0, (100.0, 400.0))
    tr.update(1, (100.0, 420.0))  # motion seen
    tr.update(2, None)  # single miss
    assert tr.update(3, (100.0, 440.0)) == "LIVE"


def test_no_rally_without_consecutive_detections():
    tr = _tracker(live_frames=3)
    tr.update(0, (100.0, 400.0))
    tr.update(1, None)
    tr.update(2, (100.0, 400.0))
    tr.update(3, None)
    assert tr.state == "IDLE"
    assert tr.events == []


def test_landing_ends_rally_without_detection_gap():
    """Shuttle stays visible on the court after landing; velocity collapse ends rally."""
    tr = _tracker(
        land_stable_frames=2,
        min_flight_frames=2,
        flight_vy_px=4.0,
        land_vy_px=2.5,
    )
    tr.update(0, (200.0, 100.0))
    tr.update(1, (200.0, 110.0))  # serve motion
    assert tr.state == "LIVE"
    tr.update(2, (200.0, 120.0))
    tr.update(3, (200.0, 125.0))
    tr.update(4, (200.0, 126.0))  # landing: low vy
    assert tr.update(5, (200.0, 126.5)) == "END"
    assert len(tr.events) == 1
    assert tr.events[0].end_frame == 5


def test_static_shuttle_on_court_does_not_start_rally():
    """Continuous detections with no motion (shuttle on ground) must not open a rally."""
    tr = _tracker()
    for i in range(20):
        tr.update(i, (300.0, 500.0))
    assert tr.state == "IDLE"
    assert tr.events == []


def test_fast_exchange_merges_rallies(monkeypatch):
    monkeypatch.setenv("SHUTTLE_RALLY_MERGE_MS", "2000")
    tr = _tracker(
        post_land_cooldown_frames=0,
        land_stable_frames=2,
        min_flight_frames=2,
        merge_ms=2000.0,
    )

    def flight_then_land(start: int) -> int:
        tr.update(start, (200.0, 100.0))
        tr.update(start + 1, (200.0, 112.0))
        tr.update(start + 2, (200.0, 124.0))
        tr.update(start + 3, (200.0, 128.0))
        tr.update(start + 4, (200.0, 128.5))
        end_frame = start + 5
        tr.update(end_frame, (200.0, 128.5))
        return end_frame

    end1 = flight_then_land(0)
    tr.update(end1 + 1, None)
    # Quick re-start within merge window (30 fps → ~0.17s gap)
    end2 = flight_then_land(end1 + 2)
    assert len(tr.events) == 1
    assert tr.events[0].start_frame == 0
    assert tr.events[0].end_frame == end2


def test_multiple_rallies_via_landing_cycles():
    tr = _tracker(
        land_stable_frames=2,
        min_flight_frames=2,
        post_land_cooldown_frames=0,
    )

    def flight_then_land(start: int) -> int:
        tr.update(start, (200.0, 100.0))
        tr.update(start + 1, (200.0, 112.0))
        tr.update(start + 2, (200.0, 124.0))
        tr.update(start + 3, (200.0, 128.0))
        tr.update(start + 4, (200.0, 128.5))
        end_frame = start + 5
        tr.update(end_frame, (200.0, 128.5))
        return end_frame

    end1 = flight_then_land(0)
    assert tr.state == "END"
    tr.update(end1 + 1, None)  # END -> IDLE

    end2 = flight_then_land(end1 + 2)
    assert tr.state == "END"
    tr.update(end2 + 1, None)

    end3 = flight_then_land(end2 + 2)
    assert tr.state == "END"
    assert len(tr.events) == 3
