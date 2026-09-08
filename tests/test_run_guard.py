import json

from fpl_pipeline import config
from fpl_pipeline.run import guard_synthetic_archive, withheld_player_columns, _PLAYER_MARKETS


def _write_manifest(tmp_path, monkeypatch, states, gw=12):
    """states: {filename: 'real'|'derived'|'synthetic'} -> a provenance manifest at a temp path."""
    path = tmp_path / "_provenance.json"
    doc = {"gw": gw, "updated": None,
           "markets": {f: {"state": s, "source": "test", "detail": "", "ts": ""} for f, s in states.items()}}
    path.write_text(json.dumps(doc))
    monkeypatch.setattr(config, "PROVENANCE_JSON", str(path))


def test_guard_withholds_only_the_synthetic_market_columns(tmp_path, monkeypatch):
    # every scraped market real EXCEPT bookings -> archive everything, but blank the yellow-card
    # columns so the synthetic carry never enters the factor history.
    states = {f: "real" for f in _PLAYER_MARKETS}
    states["sportsbet_booking_odds.csv"] = "synthetic"
    _write_manifest(tmp_path, monkeypatch, states)
    assert guard_synthetic_archive(12) == "all"
    assert withheld_player_columns() == ("Yellow Card",)
    assert withheld_player_columns(force=True) == ()
    assert guard_synthetic_archive(12, force=True) == "all"
    assert guard_synthetic_archive(None) == "none"


def test_guard_accepts_derived_markets(tmp_path, monkeypatch):
    # 2+ goals (Poisson off the goalscorer price) and GK saves (off shots-on-target) are derived
    # from THIS week's real odds - grounded, so they neither block nor get withheld.
    states = {f: "real" for f in _PLAYER_MARKETS}
    states["sportsbet_two_goals_odds.csv"] = "derived"
    states["sportsbet_goalkeeper_saves_odds.csv"] = "derived"
    _write_manifest(tmp_path, monkeypatch, states)
    assert guard_synthetic_archive(12) == "all"
    assert withheld_player_columns() == ()


def test_guard_refuses_when_manifest_is_another_gameweek(tmp_path, monkeypatch):
    # the sportsbet files are stamped GW11 (not rolled/scraped yet) -> a --gw 12 run must archive
    # nothing: it would relabel GW11 fixtures as GW12 and record last week's carry as GW12 history.
    _write_manifest(tmp_path, monkeypatch, {f: "real" for f in _PLAYER_MARKETS}, gw=11)
    assert guard_synthetic_archive(12) == "none"
    assert guard_synthetic_archive(12, force=True) == "all"


def test_guard_fixtures_only_when_everything_is_the_seed(tmp_path, monkeypatch):
    _write_manifest(tmp_path, monkeypatch, {f: "synthetic" for f in _PLAYER_MARKETS})
    assert guard_synthetic_archive(12) == "fixtures_only"


def test_guard_passes_once_all_player_markets_real(tmp_path, monkeypatch):
    _write_manifest(tmp_path, monkeypatch, {f: "real" for f in _PLAYER_MARKETS})
    assert guard_synthetic_archive(12) == "all"
    assert guard_synthetic_archive(None) == "none"


def test_guard_withholds_when_manifest_missing(tmp_path, monkeypatch):
    # no manifest -> markets read as 'unknown' (not real) -> conservative: player history withheld.
    monkeypatch.setattr(config, "PROVENANCE_JSON", str(tmp_path / "does_not_exist.json"))
    assert guard_synthetic_archive(12) == "fixtures_only"
