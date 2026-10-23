# -*- coding: utf-8 -*-
"""Train the deployable forward-projection models on 2025-26 + the current season's resolved weeks and persist them for the
pipeline to serve. Only the components where the model beats the *actual* pipeline value are
deployed (the defensive markets): clean_sheet, concede2, saves3. score1 is excluded — the
existing F1-blend already beats the model there; assist/yellow never beat the baseline.

    env/Scripts/python tools/save_projection_models.py

Writes outputs/models/proj_<stat>.joblib (the fitted LGBMRegressor) and proj_meta.joblib
(FEATURES, CATEGORICAL, the per-stat hyperparameters). Hyperparameters are the common set chosen
by the walk-forward pooled-VAL selection (tools/walkforward_projection.py)."""
import os
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fpl_pipeline import config  # noqa: E402
from tools.train_projection_model import CATEGORICAL, FIXED, SEASON  # noqa: E402
from tools.build_training_data import FEATURES  # noqa: E402

# Stats served WITH the baseline `predicted` as an extra feature. Currently EMPTY: assist was tried
# (2026-08) — `predicted` beats the RAW baseline but the deployed 0.85 blend of the MODEL loses to
# the 0.85 blend of the BASELINE (-2.1% walk-forward), so assist stays on the blended baseline. The
# per-stat-feature machinery is kept for the next candidate. `predicted` is leak-free (audited).
PREDICTED_STATS = {"score1", "yellow"}   # the stacked stats take the raw baseline as a feature
LONG_FORM = [f for f in FEATURES if f.startswith("form") and (f.endswith("6") or f.endswith("10"))]
DEFENSIVE = {"clean_sheet", "concede2", "saves3"}   # validated WITHOUT the long windows - keep that set

# Walk-forward-selected common hyperparameters per deployable component (defensive markets only —
# the attacking stats are better served by their F1-blend than by any tree; see memory).
DEPLOY = {
    "clean_sheet": dict(max_depth=3, subsample=1.0, colsample_bytree=0.6,
                        min_split_gain=0.0, min_child_weight=0.001, reg_alpha=1.0),
    "concede2": dict(max_depth=3, subsample=1.0, colsample_bytree=0.6,
                     min_split_gain=0.0, min_child_weight=5.0, reg_alpha=0.1),
    "saves3": dict(max_depth=3, subsample=1.0, colsample_bytree=0.6,
                   min_split_gain=0.0, min_child_weight=5.0, reg_alpha=0.1),
    # stacked attacking stats (config.PROJECTION_STACK): tuned 2026-09-02 on set E, pooled VAL
    "score1": dict(max_depth=5, subsample=0.6, colsample_bytree=0.5, min_split_gain=0.0,
                   min_child_weight=1.0, reg_alpha=0.0, reg_lambda=5.0, learning_rate=0.05),
    "yellow": dict(max_depth=5, subsample=0.6, colsample_bytree=0.5, min_split_gain=0.0,
                   min_child_weight=1.0, reg_alpha=0.0, reg_lambda=5.0, learning_rate=0.05),
}
VAL_WEEKS = 2   # last N prediction-weeks held out to early-stop the tree count


def features_for(stat):
    base = [f for f in FEATURES if f not in LONG_FORM] if stat in DEFENSIVE else list(FEATURES)
    return base + (["predicted"] if stat in PREDICTED_STATS else [])


def _prep(df, feats):
    X = df[feats].copy()
    for c in feats:
        if c not in CATEGORICAL:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


# 2026-10-17: train on 2025-26 AND the current season's resolved weeks. Leak-free test (2026-27 gw_to 6-7
# held out, trained on everything resolved before): clean_sheet -20.7%, concede2 -18.5%, score1 -21.6%,
# yellow -6.7% MAE vs the 2025-26-only models (player bootstrap P = 1.00 each). Validation for early
# stopping = the current season's latest resolved week. A retrain that early-stops under MIN_TREES
# (saves3: 123 keepers, a one-tree fit) keeps the existing model file instead.
TRAIN_SEASONS = ("2025-2026", config.SEASON)
MIN_TREES = 20


def train_one(stat, params):
    df = pd.read_csv(os.path.join(config.OUTPUTS_DIR, "training", f"train_{stat}.csv"))
    df = df[df["season"].isin(TRAIN_SEASONS)].dropna(subset=["actual"]).reset_index(drop=True)
    X, y = _prep(df, features_for(stat)), df["actual"].astype(float)
    cur = df["season"] == config.SEASON
    if cur.any():
        last = df.loc[cur, "gw_to"].max()
        va = cur & (df["gw_to"] == last)
        tr = ~va & ~(cur & (df["gw_to"] > last))
    else:                                   # pre-season: the old within-2025-26 split
        weeks = np.sort(df["gw_from"].unique())
        cut = weeks[-VAL_WEEKS]
        tr, va = df["gw_from"] < cut, df["gw_from"] >= cut
    p = {**FIXED, **params, "num_leaves": 2 ** params["max_depth"] - 1}
    model = lgb.LGBMRegressor(**p)
    model.fit(X[tr], y[tr], eval_X=X[va], eval_y=y[va], eval_metric="l2",   # squared error (was l1 until 2026-10-23)
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model, model.best_iteration_, int(tr.sum()), int(va.sum())


def main():
    outdir = os.path.join(config.OUTPUTS_DIR, "models")
    os.makedirs(outdir, exist_ok=True)
    print(f"{'stat':<12}{'trees':>6}{'n_train':>9}{'n_val':>7}")
    for stat, params in DEPLOY.items():
        model, trees, ntr, nva = train_one(stat, params)
        if trees < MIN_TREES and os.path.exists(os.path.join(outdir, f"proj_{stat}.joblib")):
            print(f"{stat:<12}{trees:>6}{ntr:>9}{nva:>7}  early-stopped under {MIN_TREES} trees - existing model kept")
            continue
        joblib.dump(model, os.path.join(outdir, f"proj_{stat}.joblib"))
        print(f"{stat:<12}{trees:>6}{ntr:>9}{nva:>7}")
    joblib.dump({"features": FEATURES, "categorical": CATEGORICAL, "deploy": list(DEPLOY),
                 "features_by_stat": {s: features_for(s) for s in DEPLOY}},
                os.path.join(outdir, "proj_meta.joblib"))
    print(f"\nsaved {len(DEPLOY)} models + proj_meta.joblib -> {os.path.relpath(outdir, ROOT)}")


if __name__ == "__main__":
    main()
