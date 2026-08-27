# -*- coding: utf-8 -*-
"""Train the deployable forward-projection models on ALL 2025-26 data and persist them for the
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
PREDICTED_STATS = set()

# Walk-forward-selected common hyperparameters per deployable component (defensive markets only —
# the attacking stats are better served by their F1-blend than by any tree; see memory).
DEPLOY = {
    "clean_sheet": dict(max_depth=3, subsample=1.0, colsample_bytree=0.6,
                        min_split_gain=0.0, min_child_weight=0.001, reg_alpha=1.0),
    "concede2": dict(max_depth=3, subsample=1.0, colsample_bytree=0.6,
                     min_split_gain=0.0, min_child_weight=5.0, reg_alpha=0.1),
    "saves3": dict(max_depth=3, subsample=1.0, colsample_bytree=0.6,
                   min_split_gain=0.0, min_child_weight=5.0, reg_alpha=0.1),
}
VAL_WEEKS = 2   # last N prediction-weeks held out to early-stop the tree count


def features_for(stat):
    return FEATURES + (["predicted"] if stat in PREDICTED_STATS else [])


def _prep(df, feats):
    X = df[feats].copy()
    for c in feats:
        if c not in CATEGORICAL:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


def train_one(stat, params):
    df = pd.read_csv(os.path.join(config.OUTPUTS_DIR, "training", f"train_{stat}.csv"))
    df = df[df["season"] == SEASON].dropna(subset=["actual"]).reset_index(drop=True)
    X, y = _prep(df, features_for(stat)), df["actual"].astype(float)
    weeks = np.sort(df["gw_from"].unique())
    cut = weeks[-VAL_WEEKS]
    tr, va = df["gw_from"] < cut, df["gw_from"] >= cut
    p = {**FIXED, **params, "num_leaves": 2 ** params["max_depth"] - 1}
    model = lgb.LGBMRegressor(**p)
    model.fit(X[tr], y[tr], eval_X=X[va], eval_y=y[va], eval_metric="l1",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model, model.best_iteration_, int(tr.sum()), int(va.sum())


def main():
    outdir = os.path.join(config.OUTPUTS_DIR, "models")
    os.makedirs(outdir, exist_ok=True)
    print(f"{'stat':<12}{'trees':>6}{'n_train':>9}{'n_val':>7}")
    for stat, params in DEPLOY.items():
        model, trees, ntr, nva = train_one(stat, params)
        joblib.dump(model, os.path.join(outdir, f"proj_{stat}.joblib"))
        print(f"{stat:<12}{trees:>6}{ntr:>9}{nva:>7}")
    joblib.dump({"features": FEATURES, "categorical": CATEGORICAL, "deploy": list(DEPLOY),
                 "features_by_stat": {s: features_for(s) for s in DEPLOY}},
                os.path.join(outdir, "proj_meta.joblib"))
    print(f"\nsaved {len(DEPLOY)} models + proj_meta.joblib -> {os.path.relpath(outdir, ROOT)}")


if __name__ == "__main__":
    main()
