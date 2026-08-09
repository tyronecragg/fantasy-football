import importlib.util
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("bp", os.path.join(ROOT, "tools", "build_preseason_data.py"))
bp = importlib.util.module_from_spec(spec)
sys.modules["bp"] = bp
spec.loader.exec_module(bp)


def _team(probs, team="T"):
    return pd.DataFrame({"Player": [f"P{i}" for i in range(len(probs))], "Team": team,
                         **{f"F{k}": probs for k in range(1, 7)}})


TOL = 0.01  # per-player 3-decimal rounding leaves up to ~0.005 per team


def test_normalize_scales_down_over_eleven():
    out = bp.normalize_start_probs(_team([1.0] * 12))
    assert abs(out["F1"].sum() - 11.0) < TOL
    assert (out["F1"] < 1.0).all()          # 12 "certain" players is a contradiction: all shrink


def test_normalize_down_preserves_certain_starters():
    out = bp.normalize_start_probs(_team([1.0] * 9 + [0.8, 0.7, 0.5, 0.3]))
    assert abs(out["F1"].sum() - 11.0) < TOL
    assert (out["F1"][:9] == 1.0).all()      # declared-certain players survive intact
    assert (out["F1"][9:] < [0.8, 0.7, 0.5, 0.3]).all()  # uncertainty absorbs the squeeze


def test_normalize_water_fills_up_with_cap():
    out = bp.normalize_start_probs(_team([0.9] * 10 + [0.5] * 4))  # sums to 11.0 after fill
    assert abs(out["F1"].sum() - 11.0) < TOL
    assert out["F1"].max() <= 1.0
    # the near-certain starters cap before the fringe options
    assert (out["F1"].nlargest(10) > out["F1"].nsmallest(4).max()).all()


def test_normalize_caps_thin_pool_at_certainty():
    out = bp.normalize_start_probs(_team([0.8] * 9))  # can't reach 11 even at certainty
    assert (out["F1"] == 1.0).all()


def test_depth_report_flags_short_squads(capsys):
    bp.report_pool_depth(_team([0.8] * 9))
    assert "cannot field 11" in capsys.readouterr().out


def test_normalize_does_not_mutate_input():
    df = _team([1.0] * 9 + [0.8, 0.7, 0.5, 0.3])
    bp.normalize_start_probs(df)
    assert df["F1"].iloc[9] == 0.8  # stored beliefs are never degraded


def test_normalize_is_per_team():
    df = pd.concat([_team([1.0] * 11, "A"), _team([0.5] * 14, "B")], ignore_index=True)
    out = bp.normalize_start_probs(df)
    sums = out.groupby("Team")["F1"].sum()
    assert abs(sums["A"] - 11.0) < TOL and abs(sums["B"] - 11.0) < TOL


def test_anchor_matches_market_team_goals():
    import numpy as np

    p = pd.Series([0.5, 0.3, 0.2, 0.4, 0.1])
    start = pd.Series([1.0, 1.0, 0.8, 1.0, 0.5])
    teams = pd.Series(["A", "A", "A", "B", "B"])
    anchored, scale = bp.anchor_to_team_goals(p, start, teams, {"A": 2.4, "B": 0.6})

    lam = -np.log(1 - anchored)
    sums = (start * lam).groupby(teams).sum()
    assert abs(sums["A"] - 2.4) < 1e-9 and abs(sums["B"] - 0.6) < 1e-9
    # relative ordering between teammates preserved
    assert anchored[0] > anchored[1] > anchored[2]
    # team without a market rate is untouched
    untouched, s2 = bp.anchor_to_team_goals(p, start, teams, {"A": 2.4})
    assert np.allclose(untouched[3:], p[3:]) and (s2[3:] == 1.0).all()
