"""Tests for track evidence accumulator."""

from __future__ import annotations

from track_evidence import AccumulatorConfig, TrackEvidenceBuffer


def _cfg(**kwargs) -> AccumulatorConfig:
    defaults = dict(
        min_reads=3,
        window_frames=90,
        min_spacing=5,
        high_conf_classifier=0.75,
        high_conf_ocr=0.55,
    )
    defaults.update(kwargs)
    return AccumulatorConfig(**defaults)


def test_single_read_does_not_confirm():
    buf = TrackEvidenceBuffer(config=_cfg())
    assert buf.try_add_read(10, 10, 0.90, "classifier", 10)
    assert not buf.confirmed


def test_three_spaced_reads_confirm():
    buf = TrackEvidenceBuffer(config=_cfg())
    assert buf.try_add_read(10, 10, 0.80, "classifier", 10)
    assert not buf.confirmed
    assert buf.try_add_read(16, 10, 0.82, "classifier", 10)
    assert not buf.confirmed
    assert buf.try_add_read(22, 10, 0.85, "classifier", 10)
    assert buf.confirmed
    assert buf.independent_read_count(10) == 3


def test_reads_too_close_are_not_independent():
    buf = TrackEvidenceBuffer(config=_cfg(min_reads=2, min_spacing=5))
    buf.try_add_read(10, 10, 0.80, "classifier", 10)
    buf.try_add_read(12, 10, 0.82, "classifier", 10)
    assert buf.independent_read_count(10) == 1


def test_low_confidence_rejected():
    buf = TrackEvidenceBuffer(config=_cfg())
    assert not buf.try_add_read(10, 10, 0.50, "classifier", 10)
    assert len(buf.reads) == 0


def test_classifier_target_counts_at_lower_threshold():
    buf = TrackEvidenceBuffer(config=_cfg(min_reads=2, target_number_min_conf=0.25))
    assert buf.try_add_read(10, 23, 0.30, "classifier_target", 23)
    assert buf.try_add_read(16, 23, 0.28, "classifier_target", 23)
    assert buf.confirmed


def test_wrong_number_not_stored():
    buf = TrackEvidenceBuffer(config=_cfg())
    assert not buf.try_add_read(10, 7, 0.90, "classifier", 10)
    assert len(buf.reads) == 0
