"""Classifier + OCR weighted jersey votes for player lock."""

from __future__ import annotations

from backend.app.pipeline.matcher import (
    TrackIdentityMatcher,
    combine_target_vote_conf,
)


def test_combine_agree_uses_max_weighted():
    conf = combine_target_vote_conf(
        10,
        classifier_number=10,
        classifier_conf=0.7,
        ocr_number=10,
        ocr_conf=0.5,
    )
    assert conf == max(0.7, 0.5)


def test_combine_disagree_favors_classifier(monkeypatch):
    monkeypatch.setenv("CLASSIFIER_VOTE_WEIGHT", "1.0")
    monkeypatch.setenv("CLASSIFIER_MIN_CONF", "0.5")
    conf = combine_target_vote_conf(
        10,
        classifier_number=10,
        classifier_conf=0.8,
        ocr_number=7,
        ocr_conf=0.9,
    )
    assert conf == 0.8


def test_combine_ocr_only_when_classifier_low(monkeypatch):
    monkeypatch.setenv("CLASSIFIER_MIN_CONF", "0.5")
    conf = combine_target_vote_conf(
        10,
        classifier_number=7,
        classifier_conf=0.9,
        ocr_number=10,
        ocr_conf=0.6,
    )
    assert conf == 0.6


def test_observe_number_classifier_vote_without_display_match(monkeypatch):
    monkeypatch.setenv("NUMBER_MIN_CONF", "0.4")
    monkeypatch.setenv("CLASSIFIER_MIN_CONF", "0.5")
    monkeypatch.setenv("NUMBER_LOCK_VOTES", "0.5")
    matcher = TrackIdentityMatcher(target_jersey=10, target_name="")
    matcher.observe_number(
        3,
        7,
        0.9,
        classifier_number=10,
        classifier_conf=0.75,
        ocr_number=7,
        ocr_conf=0.9,
        use_classifier_weighted_vote=True,
    )
    assert matcher.number_votes[3] == 0.75
    assert matcher.has_number_match[3] is True


def test_observe_number_legacy_ocr_only():
    matcher = TrackIdentityMatcher(target_jersey=10, target_name="")
    matcher.observe_number(3, 10, 0.6)
    assert matcher.number_votes[3] == 0.6
    matcher.observe_number(3, 7, 0.9)
    assert matcher.number_votes[3] == 0.6
