# -*- coding: utf-8 -*-
"""Tune + fit a forward-projection model per scoring component, then score it against the
pipeline baseline on held-out data.

    env/Scripts/python tools/train_projection_model.py

Reads outputs/training/train_<stat>.csv (built by tools/build_training_data.py): objective
FEATURES + `predicted` (the current pipeline's projection = the number to beat) + `actual`
(the odds-derived value that materialised = the target). We predict `actual` directly.

Model: LightGBM (a GBM — NaN-native so the Top-6 gaps and cold-start form need no imputation;
native categorical for position/venue).

DATA ROLES
  2025-2026 is split TEMPORALLY by prediction-week (gw_from), earliest weeks first, into:
    TRAIN  earliest ~60% of weeks  -> fit
    VAL    middle  ~20% of weeks   -> early stopping AND hyperparameter selection
    TEST   latest  ~20% of weeks   -> final in-season score, touched once
  GW2  the 2026-2027 rows (GW1->GW2, i.e. F2) are a separate OUT-OF-SEASON holdout — never in
       any 2025-26 split, so it's the most real-world check we have (small, F2-only).

TUNING  a random search: each candidate is trained with early stopping on VAL, and the winner
is chosen by VAL MAE — TEST/GW2 are never consulted during selection. best_iteration (the
early-stopped tree count) is passed explicitly to predict, so no split is self-graded.
"""
import os
import random
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fpl_pipeline import config  # noqa: E402
from tools.build_training_data import FEATURES  # noqa: E402
from tools.backtest_projections import STATS  # noqa: E402

SEASON, GW2_SEASON = "2025-2026", "2026-2027"
CATEGORICAL = ["position", "venue"]
TRAIN_FRAC, VAL_FRAC = 0.60, 0.20
PATIENCE, N_TRIALS, SEED = 50, 40, 0

# Fixed across all candidates (learning rate held constant, per request)
FIXED = dict(objective="regression", n_estimators=3000, learning_rate=0.03,
             subsample_freq=1, reg_lambda=1.0, random_state=0, verbose=-1)
# Searched — XGBoost-style knobs mapped to LightGBM names (gamma->min_split_gain,
# min_child_weight = min hessian sum, alpha->reg_alpha). num_leaves is tied to max_depth
# below so depth is the effective control on this leaf-wise learner.
SPACE = dict(max_depth=[3, 4, 5, 6, 8],
             subsample=[0.6, 0.8, 1.0],           # bagging_fraction
             colsample_bytree=[0.6, 0.8, 1.0],    # feature_fraction
             min_split_gain=[0.0, 0.01, 0.1],     # gamma
             min_child_weight=[1e-3, 1.0, 5.0],   # min hessian in leaf
             reg_alpha=[0.0, 0.1, 1.0])           # alpha


def _mae(pred, actual):
    m = pd.notna(pred) & pd.notna(actual)
    return float(np.abs(np.asarray(pred)[m.values] - np.asarray(actual)[m.values]).mean()) \
        if m.any() else float("nan")


def _time_split(df):
    marks = np.sort(df["gw_from"].astype(float).unique())
    tr_cut = marks[int(len(marks) * TRAIN_FRAC) - 1]
    va_cut = marks[int(len(marks) * (TRAIN_FRAC + VAL_FRAC)) - 1]
    gw = df["gw_from"].astype(float)
    return pd.Series(np.where(gw <= tr_cut, "train",
                     np.where(gw <= va_cut, "val", "test")), index=df.index)


def _prep(df):
    X = df[FEATURES].copy()
    for c in FEATURES:
        if c not in CATEGORICAL:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


def _sample_params(rng):
    p = {k: rng.choice(v) for k, v in SPACE.items()}
    p["num_leaves"] = min(2 ** p["max_depth"] - 1, 255)   # let max_depth bind
    return {**FIXED, **p}


def _fit(params, Xtr, ytr, Xva, yva):
    model = lgb.LGBMRegressor(**params)
    model.fit(Xtr, ytr, eval_X=Xva, eval_y=yva, eval_metric="l1",
              callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    return model


def _predict(model, X):
    return pd.Series(np.clip(model.predict(X, num_iteration=model.best_iteration_), 0, 1),
                     index=X.index)


def run_component(stat, path):
    full = pd.read_csv(path)
    df = full[full["season"] == SEASON].dropna(subset=["actual"]).reset_index(drop=True)
    split = _time_split(df)
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    if min(tr.sum(), va.sum(), te.sum()) == 0:
        print(f"{stat:<12} skipped (train={tr.sum()} val={va.sum()} test={te.sum()})")
        return None
    X, y = _prep(df), df["actual"].astype(float)

    rng = random.Random(SEED)
    best = None
    for _ in range(N_TRIALS):
        params = _sample_params(rng)
        model = _fit(params, X[tr], y[tr], X[va], y[va])
        val_mae = _mae(_predict(model, X[va]), y[va])          # selection metric (VAL only)
        if best is None or val_mae < best["val_mae"]:
            best = dict(params=params, model=model, val_mae=val_mae)

    # In-season TEST (latest 2025-26 weeks), scored once
    d = df[te]
    pred_te = _predict(best["model"], X[te])
    base_mae, model_mae = _mae(d["predicted"], d["actual"]), _mae(pred_te, d["actual"])

    # Out-of-season GW2 holdout (2026-27 F2), scored once
    g = full[full["season"] == GW2_SEASON].dropna(subset=["actual"]).reset_index(drop=True)
    gw2 = None
    if len(g):
        pg = _predict(best["model"], _prep(g))
        gw2 = (len(g), _mae(g["predicted"], g["actual"]), _mae(pg, g["actual"]))

    imp = (base_mae - model_mae) / base_mae * 100
    g_txt = (f"{gw2[0]:>5}{gw2[1]:>8.4f}{gw2[2]:>8.4f}{(gw2[1]-gw2[2])/gw2[1]*100:>+7.1f}%"
             if gw2 else f"{'-':>5}{'-':>8}{'-':>8}{'-':>8}")
    print(f"{stat:<12}{best['model'].best_iteration_:>5}{int(te.sum()):>6} | "
          f"{base_mae:>8.4f}{model_mae:>8.4f}{imp:>+7.1f}% | {g_txt}")
    return dict(stat=stat, best=best, test=d.assign(model=pred_te.values))


def per_horizon(results):
    print("\nper-horizon TEST MAE (base -> model):")
    print(f"{'component':<12}{'horizon':>8}{'n':>6} | {'base':>8}{'model':>8}{'Δ%':>8}")
    print("-" * 52)
    for r in results:
        d = r["test"]
        for k in sorted(d["horizon"].unique()):
            s = d[d["horizon"] == k]
            b, mo = _mae(s["predicted"], s["actual"]), _mae(s["model"], s["actual"])
            print(f"{r['stat']:<12}{f'F{int(k)+1}':>8}{len(s):>6} | "
                  f"{b:>8.4f}{mo:>8.4f}{(b - mo) / b * 100:>+7.1f}%")


def show_best(results):
    print("\nselected hyperparameters (by VAL MAE):")
    keys = list(SPACE)
    print(f"{'component':<12}" + "".join(f"{k:>17}" for k in keys))
    for r in results:
        p = r["best"]["params"]
        print(f"{r['stat']:<12}" + "".join(f"{str(p[k]):>17}" for k in keys))


def main():
    tdir = os.path.join(config.OUTPUTS_DIR, "training")
    print(f"tuning: {N_TRIALS} random trials/component, lr={FIXED['learning_rate']} fixed, "
          f"early stop on VAL (patience {PATIENCE})")
    print(f"splits: TRAIN/VAL/TEST from {SEASON} by prediction-week; "
          f"GW2 = {GW2_SEASON} F2 out-of-season holdout\n")
    print(f"{'component':<12}{'trees':>5}{'nTe':>6} | {'base':>8}{'model':>8}{'impr':>8} | "
          f"{'nGW2':>5}{'base':>8}{'model':>8}{'impr':>8}")
    print("-" * 80)
    results = []
    for stat in STATS:
        path = os.path.join(tdir, f"train_{stat}.csv")
        if os.path.exists(path):
            r = run_component(stat, path)
            if r:
                results.append(r)
    if results:
        per_horizon(results)
        show_best(results)
    print("\n(impr>0 = model beats the current pipeline baseline; GW2 is truly out-of-season)")


if __name__ == "__main__":
    main()
