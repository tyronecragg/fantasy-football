import pytest

from fpl_pipeline import config
from fpl_pipeline.run import guard_synthetic_archive


def test_guard_blocks_archiving_synthetic_odds(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SPORTSBET_DIR", str(tmp_path))
    (tmp_path / "SYNTHETIC_NOTE.txt").write_text("placeholder odds")
    with pytest.raises(SystemExit, match="synthetic"):
        guard_synthetic_archive(12)
    guard_synthetic_archive(12, force=True)  # --force-archive overrides
    guard_synthetic_archive(None)            # plain rebuild (no archiving) is never blocked


def test_guard_passes_once_odds_are_real(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SPORTSBET_DIR", str(tmp_path))
    guard_synthetic_archive(12)  # no marker file -> archiving allowed
