import importlib.util
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("build_fixtures", os.path.join(ROOT, "tools", "build_fixtures.py"))
bf = importlib.util.module_from_spec(spec)
sys.modules["build_fixtures"] = bf
spec.loader.exec_module(bf)


def test_build_window_synthetic():
    sf = pd.DataFrame({"gameweek": [1, 1, 2],
                       "home_team": ["A", "C", "B"],
                       "away_team": ["B", "D", "C"]})
    w = bf.build_window(sf, 1, window=2).set_index("Team")

    assert w.loc["A", "GW1 Opponent"] == "B" and w.loc["A", "GW1 Venue"] == "H"
    assert w.loc["B", "GW1 Opponent"] == "A" and w.loc["B", "GW1 Venue"] == "A"
    assert w.loc["B", "GW2 Opponent"] == "C" and w.loc["B", "GW2 Venue"] == "H"
    assert pd.isna(w.loc["A", "GW2 Opponent"])  # A has no GW2 fixture (blank gameweek)


def test_real_season_file_reciprocal():
    sf = pd.read_csv(os.path.join(ROOT, "inputs", "season_fixtures.csv"))
    w = bf.build_window(sf, 1).set_index("Team")
    assert len(w) == 20
    for team in w.index:
        opp, venue = w.loc[team, "GW1 Opponent"], w.loc[team, "GW1 Venue"]
        assert w.loc[opp, "GW1 Opponent"] == team          # opponent's opponent is us
        assert {venue, w.loc[opp, "GW1 Venue"]} == {"H", "A"}
