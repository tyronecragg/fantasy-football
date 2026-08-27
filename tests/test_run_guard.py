import json

from fpl_pipeline import config
from fpl_pipeline.run import guard_synthetic_archive, _PLAYER_MARKETS


def _write_manifest(tmp_path, monkeypatch, states):
    """states: {filename: 'real'|'synthetic'} -> a provenance manifest at a temp path."""
    path = tmp_path / "_provenance.json"
    doc = {"gw": 12, "updated": None,
           "markets": {f: {"state": s, "source": "test", "detail": "", "ts": ""} for f, s in states.items()}}
    path.write_text(json.dumps(doc))
    monkeypatch.setattr(config, "PROVENANCE_JSON", str(path))


def test_guard_withholds_player_history_but_keeps_match_odds(tmp_path, monkeypatch):
    # every scraped player market real EXCEPT bookings -> synthetic player odds would poison the
    # factors, but match odds can be real and are unbackfillable, so fixtures are still archived.
    states = {f: "real" for f in _PLAYER_MARKETS}
    states["sportsbet_booking_odds.csv"] = "synthetic"
    _write_manifest(tmp_path, monkeypatch, states)
    assert guard_synthetic_archive(12) == "fixtures_only"
    assert guard_synthetic_archive(12, force=True) == "all"
    assert guard_synthetic_archive(None) == "none"


def test_guard_passes_once_all_player_markets_real(tmp_path, monkeypatch):
    _write_manifest(tmp_path, monkeypatch, {f: "real" for f in _PLAYER_MARKETS})
    assert guard_synthetic_archive(12) == "all"
    assert guard_synthetic_archive(None) == "none"


def test_guard_withholds_when_manifest_missing(tmp_path, monkeypatch):
    # no manifest -> markets read as 'unknown' (not real) -> conservative: player history withheld.
    monkeypatch.setattr(config, "PROVENANCE_JSON", str(tmp_path / "does_not_exist.json"))
    assert guard_synthetic_archive(12) == "fixtures_only"
