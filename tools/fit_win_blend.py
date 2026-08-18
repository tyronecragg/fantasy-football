"""Fit the rating x outright blend and test it against the incumbent F3+ win model.

    python tools/fit_win_blend.py

Three models, two evaluations.

MODELS
  incumbent   model.win_pred — season outrights only, what serves F3-F8 today
  rating      own/opp odds-implied rating only, fitted on five seasons of OPENING rounds
  blend       both, fitted on 2025-26 where we have point-in-time outrights AND match odds

EVALUATIONS
  1. Mid-season CV (2025-26, grouped by month, never random). Plenty of rows, but the
     wrong regime: by GW16 the outright markets are half-resolved.
  2. GW1 2026-27 against the REAL pasted match odds. Only 20 team-perspectives, but it is
     genuine out-of-sample ground truth in the August regime, for these squads, after this
     summer's transfers — the one thing no amount of historical data can give us.

The blend's outright coefficients are necessarily fitted mid-season (no historical
pre-season outrights exist), so evaluation 2 is the one that decides whether it transfers.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config, model, names  # noqa: E402
from tools.build_team_ratings import DEFAULT_SEASONS, fetch, match_probs, rate  # noqa: E402


def early_season_pool(seasons=DEFAULT_SEASONS):
    """Opening five rounds of each season, with the PRIOR season's ratings attached."""
    rows, prev = [], None
    for yy in seasons:
        d = match_probs(fetch(yy))
        if prev is not None:
            e = d.head(50).copy()
            level = float(prev.nsmallest(3).mean())
            e["own"] = e["HomeTeam"].map(prev).fillna(level)
            e["opp"] = e["AwayTeam"].map(prev).fillna(level)
            rows.append(e)
        prev = rate(d)
    return pd.concat(rows, ignore_index=True)


def fit_rating_only():
    p = early_season_pool()
    X = np.c_[np.ones(len(p)), p["own"], p["opp"]]
    h, *_ = np.linalg.lstsq(X, p["p_home"].values, rcond=None)
    a, *_ = np.linalg.lstsq(X, p["p_away"].values, rcond=None)
    return h, a, len(p)


def midseason_frame():
    """2025-26 matches joined to point-in-time outrights from our own archive."""
    prev = rate(match_probs(fetch("2425")))
    cur = match_probs(fetch("2526")).assign(
        Date=pd.to_datetime(pd.read_csv(os.path.join(
            config.ROOT, "fpl_data", "football_data", "E0_2526.csv"))["Date"], dayfirst=True))
    level = float(prev.nsmallest(3).mean())

    ou = pd.read_csv(os.path.join(config.OUTPUTS_DIR, "team_gw_real_odds.csv"))
    ou = ou[["Team", "Title", "Relegation"]].dropna().groupby("Team").mean()

    d = cur.copy()
    d["own_r"] = d["HomeTeam"].map(prev).fillna(level)
    d["opp_r"] = d["AwayTeam"].map(prev).fillna(level)
    for side, pre in (("HomeTeam", "own"), ("AwayTeam", "opp")):
        for m in ("Title", "Relegation"):
            d[f"{pre}_{m}"] = d[side].map(ou[m])
    return d.dropna(subset=["own_Title", "opp_Title"]).reset_index(drop=True)


RATING = ["own_r", "opp_r"]
OUTRIGHT = ["own_Title", "own_Relegation", "opp_Title", "opp_Relegation"]


def ols(d, cols, target="p_home"):
    X = np.c_[np.ones(len(d)), d[cols].values]
    beta, *_ = np.linalg.lstsq(X, d[target].values, rcond=None)
    return beta


def cv_mae(d, cols, groups, target="p_home"):
    X, y = np.c_[np.ones(len(d)), d[cols].values], d[target].values
    errs = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        b, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        errs.append(np.abs(X[te] @ b - y[te]).mean())
    return float(np.mean(errs))


def gw1_truth():
    """Real GW1 2026-27 match odds -> de-margined win probabilities per team."""
    p = os.path.join(config.INPUTS_DIR, "gw1_match_odds.csv")
    d = pd.read_csv(p)
    ov = 1 / d["home_odds"] + 1 / d["draw_odds"] + 1 / d["away_odds"]
    d["p_home"] = (1 / d["home_odds"]) / ov
    d["p_away"] = (1 / d["away_odds"]) / ov
    d["home_team"] = names.apply_team_names(d["home_team"])
    d["away_team"] = names.apply_team_names(d["away_team"])
    return d


if __name__ == "__main__":
    model.load_coefficients()

    h_beta, a_beta, n_early = fit_rating_only()
    mid = midseason_frame()
    groups = mid["Date"].dt.to_period("M").astype(str)

    print(f"rating model: {n_early} opening-round matches, 4 season transitions")
    print(f"blend model : {len(mid)} matches from 2025-26 with point-in-time outrights\n")

    print("=" * 70)
    print("1. MID-SEASON CV (2025-26, grouped by month)")
    print("=" * 70)
    print(f"  {'rating only':<22}{cv_mae(mid, RATING, groups):>9.4f}")
    print(f"  {'outrights only':<22}{cv_mae(mid, OUTRIGHT, groups):>9.4f}")
    print(f"  {'blend':<22}{cv_mae(mid, RATING + OUTRIGHT, groups):>9.4f}")

    # Fit home and away legs separately, exactly as the rating model does. Deriving the
    # away side as (1 - home - assumed draw) would bake in a fixed draw rate and hand the
    # blend a handicap the incumbent does not carry.
    blend_home = ols(mid, RATING + OUTRIGHT, "p_home")
    blend_away = ols(mid, RATING + OUTRIGHT, "p_away")

    # ---- August ground truth ----
    ratings = pd.read_csv(os.path.join(config.INPUTS_DIR, "team_ratings.csv"))
    ratings = ratings[ratings["Season"] == config.SEASON].set_index("Team")["rating"]
    master = pd.read_csv(os.path.join(config.OUTPUTS_DIR, "13_players_master.csv"))
    ou26 = master[["Team", "Title", "Relegation", "Top 6"]].drop_duplicates("Team").set_index("Team")

    truth = gw1_truth()
    rows = []
    for t in truth.itertuples():
        for team, opp, at_home, actual in ((t.home_team, t.away_team, True, t.p_home),
                                           (t.away_team, t.home_team, False, t.p_away)):
            if team not in ratings.index or team not in ou26.index:
                continue
            rows.append({"team": team, "opp": opp, "home": at_home, "actual": actual})
    g = pd.DataFrame(rows)

    # each model's prediction for the same 20 team-perspectives
    hr = g["team"].where(g["home"], g["opp"]).map(ratings)     # home-side rating
    ar = g["opp"].where(g["home"], g["team"]).map(ratings)     # away-side rating
    X = np.c_[np.ones(len(g)), hr, ar]
    g["rating"] = np.where(g["home"], X @ h_beta, X @ a_beta)

    own = g["team"].map(ou26["Title"]), g["team"].map(ou26["Relegation"])
    opp = g["opp"].map(ou26["Title"]), g["opp"].map(ou26["Relegation"])
    hT = g["team"].where(g["home"], g["opp"]); aT = g["opp"].where(g["home"], g["team"])
    Xb = np.c_[np.ones(len(g)), hT.map(ratings), aT.map(ratings),
               hT.map(ou26["Title"]), hT.map(ou26["Relegation"]),
               aT.map(ou26["Title"]), aT.map(ou26["Relegation"])]
    g["blend"] = np.where(g["home"], Xb @ blend_home, Xb @ blend_away)

    g["incumbent"] = model.win_pred(own[0], own[1], g["team"].map(ou26["Top 6"]),
                                    opp[0], opp[1], g["opp"].map(ou26["Top 6"]),
                                    g["home"]).values

    print("\n" + "=" * 70)
    print(f"2. GW1 2026-27 vs REAL ODDS — {len(g)} team-perspectives, fully out of sample")
    print("=" * 70)
    for name in ("incumbent", "rating", "blend"):
        e = (g[name] - g["actual"]).abs()
        print(f"  {name:<22}{e.mean():>9.4f}   max {e.max():.3f}   bias {(g[name] - g['actual']).mean():+.4f}")
    print(f"  {'predict the mean':<22}{(g['actual'] - g['actual'].mean()).abs().mean():>9.4f}")

    g["err_inc"] = (g["incumbent"] - g["actual"]).abs()
    g["err_rat"] = (g["rating"] - g["actual"]).abs()
    print("\n  worst incumbent misses:")
    for r in g.nlargest(5, "err_inc").itertuples():
        print(f"    {r.team:<15}{'H' if r.home else 'A'} v {r.opp:<15} actual {r.actual:.2f}  "
              f"incumbent {r.incumbent:.2f}  rating {r.rating:.2f}")
