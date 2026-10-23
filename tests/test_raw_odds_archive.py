import json
import os

from fpl_pipeline import config, history

ROWS = ("player_name,match_id,odds_decimal\n"
        "Bukayo Saka,Arsenal vs. Everton,2.960\n"     # Betway row: fixture label
        "Declan Rice,90000002,8.38\n")                # synthetic seed row: numeric id


def _manifest(markets):
    with open(config.PROVENANCE_JSON, "w") as fh:
        json.dump({"gw": 8, "updated": None,
                   "markets": {f: {"state": s, "source": "test"} for f, s in markets.items()}}, fh)


def _setup(tmp_path, monkeypatch, markets):
    src = tmp_path / "sportsbet"; src.mkdir()
    for f in markets:
        (src / f).write_text(ROWS)
    monkeypatch.setattr(config, "PROVENANCE_JSON", str(src / "_provenance.json"))
    _manifest(markets)
    return src, tmp_path / "raw"


def test_saves_only_real_markets_without_seed_rows(tmp_path, monkeypatch):
    src, dest = _setup(tmp_path, monkeypatch, {
        "sportsbet_goalscorer_odds.csv": "real", "sportsbet_assist_odds.csv": "synthetic",
        "sportsbet_two_goals_odds.csv": "derived"})
    saved = history.archive_raw_odds("2026-2027", 8, src_dir=str(src), dest_root=str(dest))
    assert saved == {"sportsbet_goalscorer_odds.csv": 1}
    week = dest / "2026-2027" / "GW08"
    assert sorted(os.listdir(week)) == ["_provenance.json", "sportsbet_goalscorer_odds.csv"]
    text = (week / "sportsbet_goalscorer_odds.csv").read_text()
    assert "2.960" in text                       # price kept exactly as scraped
    assert "90000002" not in text                # synthetic seed row dropped
    assert set(json.loads((week / "_provenance.json").read_text())) == {"sportsbet_goalscorer_odds.csv"}


def test_rerun_overwrites_but_never_deletes(tmp_path, monkeypatch):
    src, dest = _setup(tmp_path, monkeypatch, {"sportsbet_goalscorer_odds.csv": "real",
                                               "sportsbet_assist_odds.csv": "real"})
    history.archive_raw_odds("2026-2027", 8, src_dir=str(src), dest_root=str(dest))
    (src / "sportsbet_goalscorer_odds.csv").write_text(ROWS.replace("2.960", "3.10"))
    _manifest({"sportsbet_goalscorer_odds.csv": "real", "sportsbet_assist_odds.csv": "synthetic"})
    history.archive_raw_odds("2026-2027", 8, src_dir=str(src), dest_root=str(dest))
    week = dest / "2026-2027" / "GW08"
    assert "3.10" in (week / "sportsbet_goalscorer_odds.csv").read_text()
    assert (week / "sportsbet_assist_odds.csv").exists()        # saved earlier, kept


def test_nothing_real_writes_nothing(tmp_path, monkeypatch):
    src, dest = _setup(tmp_path, monkeypatch, {"sportsbet_goalscorer_odds.csv": "synthetic"})
    assert history.archive_raw_odds("2026-2027", 8, src_dir=str(src), dest_root=str(dest)) == {}
    assert not dest.exists()
