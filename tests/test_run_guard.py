import pytest

from fpl_pipeline import config
from fpl_pipeline.run import guard_synthetic_archive


def test_guard_withholds_player_history_but_keeps_match_odds(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SPORTSBET_DIR", str(tmp_path))
    (tmp_path / "SYNTHETIC_NOTE.txt").write_text("placeholder player odds")
    # Synthetic player odds would poison the factors, but match odds can be real and
    # are unbackfillable - so fixtures are still archived.
    assert guard_synthetic_archive(12) == "fixtures_only"
    assert guard_synthetic_archive(12, force=True) == "all"
    assert guard_synthetic_archive(None) == "none"


def test_guard_passes_once_odds_are_real(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SPORTSBET_DIR", str(tmp_path))
    assert guard_synthetic_archive(12) == "all"   # no marker file
    assert guard_synthetic_archive(None) == "none"
