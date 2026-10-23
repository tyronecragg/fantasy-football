import numpy as np
import pandas as pd

from fpl_pipeline import team_views


def _inputs():
    tv = pd.DataFrame({"team": ["Leeds", "Man Utd", "Arsenal", "Spurs"],
                       "f1_opponent": ["Man Utd", "Leeds", "Spurs", "Arsenal"],
                       "f1_win": [0.39, 0.33, 0.55, 0.20], "f1_opponent_win": [0.33, 0.39, 0.20, 0.55],
                       "f2_opponent": ["Arsenal", "Spurs", "Leeds", "Man Utd"],
                       "f2_win": [np.nan, np.nan, np.nan, np.nan], "f2_opponent_win": [np.nan] * 4})
    mk = {
        "clean_sheet": pd.DataFrame({"team": ["Leeds", "Man Utd", "Arsenal"], "prob": [0.27, 0.25, 0.45]}),
        "concede": pd.DataFrame({"team": ["Leeds", "Man Utd"], "prob2": [0.35, 0.38], "prob4": [0.05, np.nan]}),
        "gk_saves": pd.DataFrame({"team": ["Leeds", "Man Utd"], "prob3": [0.55, 0.5], "prob6": [0.08, 0.07]}),
        "score1": pd.DataFrame({"player": ["DCL", "Mbeumo", "Saka"], "prob": [0.38, 0.33, 0.30]}),
        "score2": pd.DataFrame({"player": ["DCL"], "prob": [0.08]}),
        "assist": pd.DataFrame({"player": ["DCL", "Mbeumo"], "prob": [0.15, 0.22]}),
        "yellow": pd.DataFrame({"player": ["DCL"], "prob": [0.2]}),
    }
    roster = pd.DataFrame({"name": ["DCL", "Mbeumo", "Saka"], "team": ["Leeds", "Man Utd", "Arsenal"]})
    return tv, mk, roster


def test_positive_view_lifts_team_and_hurts_opponent():
    tv, mk, ro = _inputs()
    tv2, mk2, rep = team_views.shade(tv, mk, ro, {"Leeds": 0.12, "Man Utd": -0.10}, weight=0.5)
    assert rep == {"Leeds": 0.06, "Man Utd": -0.05}
    assert tv2.loc[0, "f1_win"] > 0.39 and tv2.loc[1, "f1_win"] < 0.33
    assert np.isclose(tv2.loc[0, "f1_win"], tv2.loc[1, "f1_opponent_win"])      # one match, one answer
    assert np.isclose(tv2.loc[0, "f1_win"] + tv2.loc[0, "f1_opponent_win"], 0.72)  # draw kept
    cs = mk2["clean_sheet"].set_index("team")["prob"]
    assert cs["Leeds"] > 0.27 and cs["Man Utd"] < 0.25 and np.isclose(cs["Arsenal"], 0.45)
    c = mk2["concede"].set_index("team")
    assert c.loc["Leeds", "prob2"] < 0.35 and c.loc["Man Utd", "prob2"] > 0.38
    assert np.isnan(c.loc["Man Utd", "prob4"])                                   # unpriced stays unpriced
    s1 = mk2["score1"].set_index("player")["prob"]
    assert s1["DCL"] > 0.38 and s1["Mbeumo"] < 0.33 and np.isclose(s1["Saka"], 0.30)
    assert mk2["yellow"] is mk["yellow"]                                         # bookings untouched
    for d in mk2.values():
        for col in ("prob", "prob2", "prob4", "prob3", "prob6"):
            if col in d:
                v = d[col].dropna()
                assert ((v >= 0) & (v <= 1)).all()


def test_full_strength_moves_more_than_half():
    tv, mk, ro = _inputs()
    half = team_views.shade(tv, mk, ro, {"Leeds": 0.12}, weight=0.5)[1]["clean_sheet"]["prob"][0]
    full = team_views.shade(tv, mk, ro, {"Leeds": 0.12}, weight=1.0)[1]["clean_sheet"]["prob"][0]
    assert 0.27 < half < full


def test_inputs_untouched_and_noop_cases():
    tv, mk, ro = _inputs()
    before = mk["clean_sheet"].copy()
    team_views.shade(tv, mk, ro, {"Leeds": 0.12}, weight=0.5)
    pd.testing.assert_frame_equal(mk["clean_sheet"], before)
    assert team_views.shade(tv, mk, ro, {}, weight=0.5)[1] is mk
    assert team_views.shade(tv, mk, ro, {"Leeds": 0.12}, weight=0)[1] is mk


def test_unpriced_f2_win_stays_nan():
    tv, mk, ro = _inputs()
    tv2 = team_views.shade(tv, mk, ro, {"Leeds": 0.12}, weight=0.5)[0]
    assert tv2["f2_win"].isna().all()
