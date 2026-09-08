# -*- coding: utf-8 -*-
"""Prove the runtime serving path (fpl_pipeline.projection_serving) reproduces the training-time
features/predictions with zero skew. Simulate a LIVE gameweek by hiding it from the archive:

  - pick M = 27 (2025-26); truncate the archive to <= M-1 so M is 'not yet archived'
  - build a master-like frame from the archive's real M snapshot (F1 values, season odds,
    F{k} opponent/venue + opponent season odds) exactly as the pipeline master would carry
  - serve predictions for each horizon, and compare to applying the saved model to the
    training-built features for the same (player, gw_from=M, horizon) rows.

If max abs diff ~ 0, the serve-time feature reconstruction matches training.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fpl_pipeline import projection_serving as ps  # noqa: E402
from fpl_pipeline import rating as rating_mod  # noqa: E402
from tools.backtest_projections import STATS, load_archive, team_probs_by_gw  # noqa: E402
from tools.rating_backtest import gw_dates  # noqa: E402
from tools.train_projection_model import _prep  # noqa: E402

SEASON, M = "2025-2026", 27


def build_masterlike(archive, season, m):
    """Reconstruct the pipeline-master columns the server reads, from the archive's GW-m snapshot."""
    a = archive[(archive["Season"] == season) & (archive["Gameweek"] == m)].copy()
    tp = team_probs_by_gw(archive[archive["Season"] == season])
    season_odds = tp[m]                                             # team -> Title/Relegation/Top 6 at m
    out = pd.DataFrame({
        "Player Name": a["Player Name"].values, "Position": a["Position"].values, "Team": a["Team"].values,
        "Title": a["Title"].values, "Relegation": a["Relegation"].values, "Top 6": a["Top 6"].values,
    })
    for s in STATS:
        out[STATS[s]] = a[STATS[s]].values                          # F1 <stat> = persistence
    out["F1 Win"], out["F1 Opponent Win"] = a["F1 Win"].values, a["F1 Opponent Win"].values
    for k in range(2, 9):                                           # F2..F8 opponent/venue + opp odds
        opp = a.get(f"F{k} Opponent")
        out[f"F{k} Opponent"] = opp.values if opp is not None else np.nan
        out[f"F{k} Venue"] = a.get(f"F{k} Venue").values if a.get(f"F{k} Venue") is not None else np.nan
        od = season_odds.reindex(out[f"F{k} Opponent"])
        out[f"F{k} Opponent Title"] = od["Title"].values
        out[f"F{k} Opponent Relegation"] = od["Relegation"].values
        out[f"F{k} Opponent Top 6"] = od["Top 6"].values
    return out


def main():
    archive = load_archive()
    truncated = archive[~((archive["Season"] == SEASON) & (archive["Gameweek"] >= M))].copy()
    master = build_masterlike(archive, SEASON, M)
    src = rating_mod.RatingSource(refresh=False)                  # the rating as of GW M's deadline
    fr = src.at(gw_dates(archive, src.m, SEASON)[M], SEASON)
    server = ps.make_server(master, truncated, SEASON, M, rating=fr)
    if server is None:
        print("no models found — run tools/save_projection_models.py first")
        return

    # training-built features/preds for the same rows
    train = {s: pd.read_csv(os.path.join("outputs", "training", f"train_{s}.csv")) for s in server.models}

    print(f"skew check at LIVE M={M} ({SEASON}) — serve vs training model output")
    print(f"{'stat':<12}{'horizon':>8}{'n':>6}{'max|Δpred|':>12}{'mean|Δpred|':>13}")
    for k in range(2, 9):
        for s in server.models:                                     # raw baseline = training `predicted`
            t = train[s]
            t = t[(t["season"] == SEASON) & (t["gw_from"] == M) & (t["horizon"] == k - 1)]
            master[f"F{k} {STATS[s][3:]}"] = master["Player Name"].map(
                t.drop_duplicates("player").set_index("player")["predicted"])
        for s, model in server.models.items():
            served = server.predict_horizon(master, k)[s]
            served.index = master["Player Name"].values
            t = train[s]
            t = t[(t["season"] == SEASON) & (t["gw_from"] == M) & (t["horizon"] == k - 1)]
            if t.empty:
                continue
            feats = server._feats.get(s, server.FEATURES)            # each model's own feature list
            X = t[feats].copy()
            for c in feats:
                X[c] = X[c].astype("category") if c in server.CATEGORICAL else pd.to_numeric(X[c], errors="coerce")
            ref = pd.Series(np.clip(model.predict(X), 0, 1), index=t["player"].values)
            j = pd.DataFrame({"srv": served}).join(pd.DataFrame({"ref": ref}), how="inner").dropna()
            d = (j["srv"] - j["ref"]).abs()
            print(f"{s:<12}{f'F{k}':>8}{len(j):>6}{d.max():>12.6f}{d.mean():>13.6f}")


if __name__ == "__main__":
    main()
