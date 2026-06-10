"""Rally state machine transitions and win attribution."""

from backend.app.pipeline.badminton.rally_tracker import RallyTracker


def _tracker(**kwargs) -> RallyTracker:
    defaults = dict(net_y_px=360.0, fps=30.0, live_frames=2, gap_frames=5)
    defaults.update(kwargs)
    return RallyTracker(**defaults)


def test_idle_to_live_to_end_transitions():
    tr = _tracker()
    assert tr.update(0, (100.0, 400.0)) == "IDLE"
    assert tr.update(1, (105.0, 410.0)) == "LIVE"
    for i in range(2, 6):
        assert tr.update(i, (110.0, 420.0)) == "LIVE"
    state = "LIVE"
    for i in range(6, 11):
        state = tr.update(i, None)
    assert state == "END"
    assert tr.update(11, None) == "IDLE"
    assert len(tr.events) == 1
    ev = tr.events[0]
    assert ev.start_frame == 0
    assert ev.end_frame == 5


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
    tr.update(1, (100.0, 500.0), locked_foot_px=(110.0, 510.0))
    for i in range(2, 8):
        tr.update(i, None)
    assert tr.events[0].locked_player_touched_last is True

    tr2 = _tracker(touch_radius_px=50.0)
    tr2.update(0, (100.0, 500.0), locked_foot_px=(800.0, 100.0))
    tr2.update(1, (100.0, 500.0), locked_foot_px=(800.0, 100.0))
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


def test_no_rally_without_consecutive_detections():
    tr = _tracker(live_frames=3)
    tr.update(0, (100.0, 400.0))
    tr.update(1, None)
    tr.update(2, (100.0, 400.0))
    tr.update(3, None)
    assert tr.state == "IDLE"
    assert tr.events == []
