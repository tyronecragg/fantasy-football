# -*- coding: utf-8 -*-
"""Fit the per-horizon rating-gap -> (home, draw, away) maps that serve fpl_pipeline/rating.py.

    env/Scripts/python tools/fit_rating_maps.py [--seasons 2122 2223 2324 2425 2526]

For every match of the fitted seasons (the first is history only) and every horizon 1-8 weeks, the
rating gap at (match date - 7k days) is computed from prices before that cutoff, and a multinomial
logistic on [gap, |gap|, gap^2] is fitted to the match's closing probabilities (soft targets).
Writes fpl_pipeline/data/rating_maps.json. Re-run after each completed season (the maps describe
how much ratings drift per week, which is stable; the live strengths themselves are solved fresh
on every pipeline run). The evaluation/tuning harness is tools/rolling_rating.py.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fpl_pipeline import config, rating  # noqa: E402

# Seasons the maps are fitted on. The player archive (training rows) starts in 2025-26, so 2025-26
# is EXCLUDED: the maps must be out-of-sample for every season the GBMs train on, or the rating
# feature on those rows would carry a sliver of its own season. Add 2526 when the archive seasons
# move on (i.e. when 2025-26 is no longer a training season).
DEFAULT_SEASONS = ["2122", "2223", "2324", "2425"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    args = ap.parse_args()

    m = rating.load_matches(args.seasons)
    teams = rating.season_teams(m)
    seasons = sorted(teams)
    fit_seasons = seasons[1:]
    rows, grows = [], []
    cache, gcache = {}, {}
    for k in rating.HORIZONS:
        for r in m[m["season"].isin(fit_seasons)].itertuples():
            cutoff = r.date - pd.Timedelta(days=7 * k)
            key = (cutoff, r.season)
            if key not in cache:
                cache[key] = rating.ratings_at(m, cutoff, r.season, config.RATING_HALFLIFE_DAYS,
                                               config.RATING_PROMOTED_WEIGHT, teams)
                gcache[key] = rating.goals_at(m, cutoff, r.season, config.RATING_HALFLIFE_DAYS,
                                              config.RATING_PROMOTED_WEIGHT, teams)
            snap = cache[key]
            if snap is None:
                continue
            rt = snap[0]
            rows.append({"horizon": k, "diff": rt[r.HomeTeam] - rt[r.AwayTeam],
                         "p_home": r.p_home, "p_draw": r.p_draw, "p_away": r.p_away})
            gs = gcache[key]
            if gs is not None:
                att, dfn, c_h, c_a = gs
                grows.append({"horizon": k, "pred": att[r.HomeTeam] - dfn[r.AwayTeam] + c_h, "actual": np.log(r.lam_home)})
                grows.append({"horizon": k, "pred": att[r.AwayTeam] - dfn[r.HomeTeam] + c_a, "actual": np.log(r.lam_away)})
    rows, grows = pd.DataFrame(rows), pd.DataFrame(grows)

    doc = {"fitted_on": fit_seasons, "features": ["diff", "abs(diff)", "diff^2"],
           "classes": ["home", "draw", "away"],
           "params": {"halflife_days": config.RATING_HALFLIFE_DAYS,
                      "promoted_weight": config.RATING_PROMOTED_WEIGHT},
           "horizons": {}}
    print(f"{len(rows)} match-horizons from {fit_seasons}")
    for k in rating.HORIZONS:
        d = rows[rows["horizon"] == k]
        X = np.vstack([rating.features(d["diff"])] * 3)
        y = np.repeat([0, 1, 2], len(d))
        w = np.concatenate([d["p_home"].values, d["p_draw"].values, d["p_away"].values])
        clf = LogisticRegression(C=100.0, max_iter=2000).fit(X, y, sample_weight=w)
        p = clf.predict_proba(rating.features(d["diff"]))
        mae = float(np.mean(np.abs(np.concatenate([p[:, 0] - d["p_home"], p[:, 2] - d["p_away"]]))))
        doc["horizons"][str(k)] = {"coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(),
                                   "n": int(len(d)), "fit_mae": round(mae, 4)}
        print(f"  {k} week(s) ahead: n={len(d)}  in-sample win MAE {mae:.4f}")

    # goals: per-horizon linear shrink of predicted log expected-goals toward the mean
    doc["goals"] = {}
    print("attack/defence split -> expected goals (per side), in-sample:")
    for k in rating.HORIZONS:
        d = grows[grows["horizon"] == k]
        b, a = np.polyfit(d["pred"], d["actual"], 1)
        fit = np.exp(a + b * d["pred"])
        act = np.exp(d["actual"])
        naive = float(np.mean(np.abs(act - act.mean())))
        doc["goals"][str(k)] = {"a": float(a), "b": float(b), "n": int(len(d)),
                                "mae": round(float(np.mean(np.abs(fit - act))), 4), "naive_mae": round(naive, 4)}
        print(f"  {k} week(s) ahead: slope {b:.2f}  MAE {np.mean(np.abs(fit - act)):.3f} goals (predict-the-mean {naive:.3f})")
    os.makedirs(os.path.dirname(config.RATING_MAPS_JSON), exist_ok=True)
    with open(config.RATING_MAPS_JSON, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    print(f"wrote {os.path.relpath(config.RATING_MAPS_JSON, ROOT)}")


if __name__ == "__main__":
    main()
