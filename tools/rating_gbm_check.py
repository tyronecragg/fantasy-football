# -*- coding: utf-8 -*-
"""Do the deployed GBMs gain from the rolling rating's win / opponent-win as features?

    env/Scripts/python tools/rating_gbm_check.py [--trials 20] [--stats clean_sheet,concede2,saves3]

Needs outputs/training/train_<stat>.csv built with RATING_FEATURES=1 (columns rating_win/rating_opp).
Same leak-free walk-forward folds as tools/walkforward_projection.py, one common hyperparameter set
per (stat, feature set) chosen by pooled VAL MAE, then a PAIRED BOOTSTRAP over players on the pooled
TEST rows so the verdict is a probability, not a third decimal. Also reports the per-horizon split.
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
from tools.build_training_data import FEATURES  # noqa: E402
from tools.train_projection_model import CATEGORICAL, SEASON, _fit, _mae, _sample_params  # noqa: E402
from tools.walkforward_projection import make_folds  # noqa: E402

# compare the currently deployed feature set (everything but the attack/defence expected goals)
# against the full set - i.e. do the GBMs gain from rating_xg_for / rating_xg_against?
RATING = [f for f in FEATURES if f.startswith("rating_xg")]
BASE = [f for f in FEATURES if f not in RATING]


def prep(df, feats):
    X = df[feats].copy()
    for c in feats:
        X[c] = X[c].astype("category") if c in CATEGORICAL else pd.to_numeric(X[c], errors="coerce")
    return X


def run(df, feats, cands, folds):
    X, y = prep(df, feats), df["actual"].astype(float)
    scores = []
    for p in cands:
        pr, ac = [], []
        for _, tr, va, te in folds:
            m = _fit(p, X[tr], y[tr], X[va], y[va])
            pr.append(pd.Series(np.clip(m.predict(X[va], num_iteration=m.best_iteration_), 0, 1)))
            ac.append(y[va].reset_index(drop=True))
        scores.append(_mae(pd.concat(pr, ignore_index=True), pd.concat(ac, ignore_index=True)))
    best = cands[int(np.argmin(scores))]
    out = []
    for T, tr, va, te in folds:
        m = _fit(best, X[tr], y[tr], X[va], y[va])
        out.append(df[te].assign(model=np.clip(m.predict(X[te], num_iteration=m.best_iteration_), 0, 1), fold=T))
    return pd.concat(out, ignore_index=True), best


def bootstrap(a, b, actual, player, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    d = (a - actual).abs() - (b - actual).abs()          # >0 where b is better
    byp = d.groupby(player).agg(["sum", "size"])
    boots = np.array([byp["sum"].values[i].sum() / byp["size"].values[i].sum()
                      for i in (rng.integers(0, len(byp), len(byp)) for _ in range(n))])
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), float((boots > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--stats", default="clean_sheet,concede2,saves3")
    args = ap.parse_args()
    rng = random.Random(0)
    cands = [_sample_params(rng) for _ in range(args.trials)]
    tdir = os.path.join(config.OUTPUTS_DIR, "training")
    for stat in args.stats.split(","):
        df = pd.read_csv(os.path.join(tdir, f"train_{stat}.csv"))
        assert all(c in df.columns for c in RATING), "rebuild training data (tools/build_training_data.py)"
        df = df[df["season"] == SEASON].dropna(subset=["actual"]).reset_index(drop=True)
        folds = make_folds(df)
        base, pb = run(df, BASE, cands, folds)
        plus, pp = run(df, BASE + RATING, cands, folds)
        key = ["player", "gw_from", "gw_to"]
        j = base.merge(plus[key + ["model"]], on=key, suffixes=("", "_r"))
        gain, lo, hi, p = bootstrap(j["model"], j["model_r"], j["actual"], j["player"])
        print(f"\n{stat}: {len(j)} test rows, {len(folds)} folds")
        print(f"  pooled TEST MAE  deployed set {_mae(j['model'], j['actual']):.4f}  + xg {_mae(j['model_r'], j['actual']):.4f}"
              f"  (baseline {_mae(j['predicted'], j['actual']):.4f})")
        print(f"  paired bootstrap over players: gain {gain:+.5f}  95% [{lo:+.5f}, {hi:+.5f}]  P(+xg better) = {p:.2f}")
        for k in sorted(j["horizon"].unique()):
            s = j[j["horizon"] == k]
            print(f"    F{int(k)+1:<2} n={len(s):>5}  deployed {_mae(s['model'], s['actual']):.4f}  +xg {_mae(s['model_r'], s['actual']):.4f}")


if __name__ == "__main__":
    main()
