"""SAM sport-aware defaults."""

from backend.app.pipeline.segmentation import sam_enabled_for_sport


def test_sam_default_off_for_badminton(monkeypatch):
    monkeypatch.delenv("SAM_ENABLED", raising=False)
    assert sam_enabled_for_sport(is_badminton=True) is False


def test_sam_default_on_for_football(monkeypatch):
    monkeypatch.delenv("SAM_ENABLED", raising=False)
    assert sam_enabled_for_sport(is_badminton=False) is True


def test_sam_explicit_enable_for_badminton(monkeypatch):
    monkeypatch.setenv("SAM_ENABLED", "1")
    assert sam_enabled_for_sport(is_badminton=True) is True
