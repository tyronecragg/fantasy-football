"""betway.py two-gameweek scrape: the current gameweek fills the F1 files, the next fills
the *_f2 files, and player props are never scraped for F2 (the pipeline model-projects
them). Also guards the team-goals column order that concede_market() reads positionally.
"""
import importlib.util
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("betway", os.path.join(ROOT, "tools", "betway.py"))
betway = importlib.util.module_from_spec(spec)
sys.modules["betway"] = betway
spec.loader.exec_module(betway)

from fpl_pipeline import markets  # noqa: E402

# Betway's Total Goals is the MATCH total; sbv carries the line, selection is Over/Under.
_CANNED = {
    "e1": [("Total Goals", "Over", "1.5", 1.30), ("Total Goals", "Under", "1.5", 3.50),
           ("Total Goals", "Over", "3.5", 3.00), ("Total Goals", "Under", "3.5", 1.36),
           ("Arsenal To Keep A Clean Sheet", "Yes", "", 1.90),
           ("Arsenal To Keep A Clean Sheet", "No", "", 1.80),
           ("Aston Villa To Keep A Clean Sheet", "Yes", "", 3.20),
           ("Aston Villa To Keep A Clean Sheet", "No", "", 1.30),
           ("Anytime Goalscorer", "Saka, Bukayo", "", 2.50),
           ("Player 1+ Assists", "Saka, Bukayo", "", 3.50)],
    "e2": [("Total Goals", "Over", "1.5", 1.28), ("Total Goals", "Under", "1.5", 3.60),
           ("Total Goals", "Over", "3.5", 2.90), ("Total Goals", "Under", "3.5", 1.40),
           ("Chelsea To Keep A Clean Sheet", "Yes", "", 1.95),
           ("Chelsea To Keep A Clean Sheet", "No", "", 1.75),
           ("Everton To Keep A Clean Sheet", "Yes", "", 3.40),
           ("Everton To Keep A Clean Sheet", "No", "", 1.28),
           ("Anytime Goalscorer", "Palmer, Cole", "", 2.10)],   # must NOT reach the F1 files
}
_ROWS = [("e1", "Arsenal vs. Aston Villa", "2026-08-21T19:00:00"),
         ("e2", "Chelsea vs. Everton", "2026-08-28T19:00:00")]
_GW_OF = {"Arsenal vs. Aston Villa": "f1", "Chelsea vs. Everton": "f2"}


def _collect(monkeypatch):
    monkeypatch.setattr(betway, "markets_for", lambda event_id, **kw: event_id)
    monkeypatch.setattr(betway, "selections", lambda payload: _CANNED[payload])
    monkeypatch.setattr(betway.time, "sleep", lambda *a, **k: None)
    out, _ = betway.collect(_ROWS, _GW_OF, delay=0)
    return out


def test_assign_f1_f2_splits_first_ten_next_ten():
    fake = [(f"x{i}", f"M{i}", f"2026-08-{21 if i < 10 else 28}T19:00:00") for i in range(20)]
    gw_of, ordered = betway.assign_f1_f2(fake)
    assert sum(v == "f1" for v in gw_of.values()) == 10
    assert sum(v == "f2" for v in gw_of.values()) == 10
    assert [m for _, m, _ in ordered][:10] == [f"M{i}" for i in range(10)]   # F1 block first


def test_only_one_gameweek_leaves_f2_empty():
    ten = [(f"x{i}", f"M{i}", "2026-08-21T19:00:00") for i in range(10)]
    gw_of, ordered = betway.assign_f1_f2(ten)
    assert all(v == "f1" for v in gw_of.values())
    assert len(ordered) == 10


def test_f2_player_props_never_scraped(monkeypatch):
    out = _collect(monkeypatch)
    assert set(out["score1"]["match_id"]) == {"Arsenal vs. Aston Villa"}
    assert "Cole Palmer" not in set(out["score1"]["player_name"])


def test_team_markets_routed_to_f2_files(monkeypatch):
    out = _collect(monkeypatch)
    assert len(out["team_goals"]) == 2 and len(out["team_goals_f2"]) == 2      # both perspectives
    assert len(out["clean_sheet"]) == 2 and len(out["clean_sheet_f2"]) == 2
    assert set(out["clean_sheet_f2"]["team_name"]) == {"Chelsea", "Everton"}


def test_team_goals_column_order_matches_concede_market(monkeypatch):
    out = _collect(monkeypatch)
    cols = list(out["team_goals"].columns)
    # concede_market reads cols 4-5 as the 2+ line and 6-7 as the 4+ line, positionally.
    assert cols[4:8] == ["Team_Over_1.5", "Team_Under_1.5", "Team_Over_3.5", "Team_Under_3.5"]
    cm = markets.concede_market(out["team_goals"])
    assert {"Arsenal", "Aston Villa"} <= set(cm["team"])          # both sides, not just away
    assert (cm["prob4"] < cm["prob2"]).all()                     # 4+ rarer than 2+ (bug: equal)
