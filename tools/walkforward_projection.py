# -*- coding: utf-8 -*-
"""Walk-forward validation of the forward-projection models, with a single COMMON
hyperparameter set per component chosen by pooled validation.

    env/Scripts/python tools/walkforward_projection.py

The in-season test in tools/train_projection_model.py is only ~3 weeks. This slides the
train/val/test window across the whole 2025-26 season so every back-half week gets to be a
test week exactly once — the effective test set becomes the season's back half, all real
(non-synthetic) data.

LEAK-FREE FOLDS. Each fold stands at a "current week" T:
    TRAIN  rows whose outcome was resolved by then   (gw_to <= T-1)
    VAL    the just-resolved week                     (gw_to == T)      early stopping
    TEST   predictions MADE at week T for the future  (gw_from == T)    scored, outcomes > T
Training never sees an outcome the test rows are forecasting. Each TEST fold spans the full
F2-F8 horizon range.

TUNING. A fixed pool of random candidates (same pool for every component and fold). For each
candidate we run every fold and pool the VAL MAE; the candidate with the best pooled VAL MAE
is the component's COMMON set. It is then re-run across folds and its pooled TEST MAE is
compared to the pipeline baseline. TEST is never consulted during selection.
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fpl_pipeline import config  # noqa: E402
from tools.backtest_projections import STATS  # noqa: E402
from tools.train_projection_model import (  # noqa: E402
    SEASON, SPACE, _fit, _mae, _predict, _prep, _sample_params)

N_TRIALS, SEED, MIN_TRAIN = 30, 0, 300


def make_folds(df, embargo=0):
    """(T, train_mask, val_mask, test_mask) per current-week T with enough resolved history.

    embargo: weeks of gap held out between the newest TRAIN label and VAL/TEST. The backward
    10-GW form window makes val rows serially correlated with the most recent train weeks; a
    gap de-correlates the early-stopping/selection signal. TRAIN keeps gw_to <= T-1-embargo,
    so its labels are all at least `embargo` weeks older than val's (gw_to == T)."""
    gw_from, gw_to = df["gw_from"].astype(float), df["gw_to"].astype(float)
    out = []
    for T in sorted(gw_from.unique()):
        tr, va, te = (gw_to <= T - 1 - embargo), (gw_to == T), (gw_from == T)
        if tr.sum() >= MIN_TRAIN and va.sum() and te.sum():
            out.append((int(T), tr.values, va.values, te.values))
    return out


def pooled_val_mae(params, X, y, folds):
    preds, acts = [], []
    for _, tr, va, te in folds:
        model = _fit(params, X[tr], y[tr], X[va], y[va])
        preds.append(_predict(model, X[va]))
        acts.append(y[va])
    return _mae(pd.concat(preds, ignore_index=True), pd.concat(acts, ignore_index=True))


def eval_on_test(params, df, X, y, folds):
    frames = []
    for T, tr, va, te in folds:
        model = _fit(params, X[tr], y[tr], X[va], y[va])
        pt = _predict(model, X[te])
        frames.append(df[te].assign(model=pt.values, fold=T))
    return pd.concat(frames, ignore_index=True)


def run_component(stat, path, candidates, embargo=0):
    df = pd.read_csv(path)
    df = df[df["season"] == SEASON].dropna(subset=["actual"]).reset_index(drop=True)
    folds = make_folds(df, embargo)
    if not folds:
        print(f"{stat:<12} skipped (no usable folds)")
        return None
    X, y = _prep(df), df["actual"].astype(float)

    scores = [pooled_val_mae(p, X, y, folds) for p in candidates]     # selection on VAL
    best = candidates[int(np.argmin(scores))]
    test = eval_on_test(best, df, X, y, folds)                        # score on TEST

    base, model = _mae(test["predicted"], test["actual"]), _mae(test["model"], test["actual"])
    print(f"{stat:<12}{len(folds):>5}{len(test):>7} | {base:>8.4f}{model:>8.4f}"
          f"{(base - model) / base * 100:>+7.1f}%")
    return dict(stat=stat, test=test, best=best, folds=folds)


def per_horizon(results):
    print("\nper-horizon pooled TEST MAE (base -> model):")
    print(f"{'component':<12}{'horizon':>8}{'n':>6} | {'base':>8}{'model':>8}{'Δ%':>8}")
    print("-" * 52)
    for r in results:
        for k in sorted(r["test"]["horizon"].unique()):
            s = r["test"][r["test"]["horizon"] == k]
            b, mo = _mae(s["predicted"], s["actual"]), _mae(s["model"], s["actual"])
            print(f"{r['stat']:<12}{f'F{int(k)+1}':>8}{len(s):>6} | "
                  f"{b:>8.4f}{mo:>8.4f}{(b - mo) / b * 100:>+7.1f}%")


def show_best(results):
    keys = list(SPACE)
    print("\ncommon hyperparameters per component (by pooled VAL MAE):")
    print(f"{'component':<12}" + "".join(f"{k:>17}" for k in keys))
    for r in results:
        print(f"{r['stat']:<12}" + "".join(f"{str(r['best'][k]):>17}" for k in keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embargo", type=int, default=0,
                    help="weeks of gap held out between newest TRAIN label and VAL/TEST")
    args = ap.parse_args()

    rng = random.Random(SEED)
    candidates = [_sample_params(rng) for _ in range(N_TRIALS)]       # same pool for all components
    tdir = os.path.join(config.OUTPUTS_DIR, "training")
    print(f"walk-forward on {SEASON}: leak-free folds, embargo={args.embargo}w, "
          f"{N_TRIALS} candidate sets, common set per component by pooled VAL\n")
    print(f"{'component':<12}{'folds':>5}{'nTest':>7} | {'base':>8}{'model':>8}{'impr':>8}")
    print("-" * 48)
    results = []
    for stat in STATS:
        path = os.path.join(tdir, f"train_{stat}.csv")
        if os.path.exists(path):
            r = run_component(stat, path, candidates, args.embargo)
            if r:
                results.append(r)
    if results:
        per_horizon(results)
        show_best(results)
    print("\n(impr>0 = model beats the pipeline baseline, pooled over the season's back half)")


if __name__ == "__main__":
    main()
