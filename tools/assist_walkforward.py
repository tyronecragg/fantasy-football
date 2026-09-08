# -*- coding: utf-8 -*-
"""Assist forward-projection experiment: can a LightGBM beat the deployed F1-blend for F2-F8 assists?

    env/Scripts/python tools/assist_walkforward.py [--trials 30] [--sets A,B,C,D] [--stat assist]

Differences from tools/walkforward_projection.py (which said "assist never wins"):
  * LIKE-FOR-LIKE SCORING. The pipeline baseline (`predicted`) does not exist before GW21 of 2025-26
    (Top-6 odds were not archived), so the old script scored the model on ~17k rows and the baseline
    on ~10k. Here every comparison is on the rows where BOTH exist.
  * GROWING TRAIN + RETAINED LEARNINGS. Folds walk forward week by week; train = every resolved
    outcome so far, val = the just-resolved week (early stopping), test = forecasts made this week.
    One common hyperparameter set is chosen by pooled VAL MAE; each fold's early-stopped tree count
    is then REUSED to refit on train+val (so the final model never throws its validation week away).
  * FEATURE SETS, added cumulatively:
      A  the 19 objective odds features the deployed models use
      B  + fixture context: win_hat / opp_hat for the target fixture and `predicted` (the baseline
           itself, so the tree starts from parity and only has to learn corrections)
      C  + longer odds-form windows (6, 10 weeks) and their fixture-difficulty context
      D  + real on-pitch xA: trailing xA (3/6/10), xA per 90, minutes, real assists, xA MINUS the
           odds-implied probability that week (is he out-/under-performing his price?), xA split by
           opponent strength (vs stronger / weaker sides), and the short-minus-long xA trend
  * PER-HORIZON TABLE (F2 .. F8) and a paired bootstrap over players for the pooled verdict.
  * STACK. A one-parameter stack w*model + (1-w)*blended_baseline fitted on VAL folds only, scored
    on TEST - often the honest way a weak-but-different model adds value.
"""
import argparse
import os
import random
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fpl_pipeline import config  # noqa: E402
from tools.backtest_projections import STATS, build_pairs, load_archive  # noqa: E402
from tools.build_training_data import _dense_trailing, add_diffs, add_form, load_real_gw  # noqa: E402
from tools.rating_backtest import rating_win_fn  # noqa: E402
from tools.train_projection_model import FIXED, SPACE, _sample_params  # noqa: E402

CATEGORICAL = ["position", "venue"]
PATIENCE, MIN_TRAIN, SEED = 50, 300, 0
BASE = ["persistence", "momentum", "form3", "form_n3", "form_win3", "form_loss3",
        "own_title", "own_releg", "own_top6", "opp_title", "opp_releg", "opp_top6",
        "title_diff", "releg_diff", "top6_diff", "strength_diff", "position", "venue", "horizon"]
# Since 2026-09-02 the pipeline's fixture strength IS the rolling rating, so `predicted` here is the
# rating-fed baseline (what the deployed blend uses) and the fixture context is rating_win/opp.
FIX = ["rating_win", "rating_opp", "predicted"]
WIN = ["form6", "form_n6", "form10", "form_n10", "form_win10", "form_loss10"]
XA = ["xa3", "xa6", "xa10", "xa_n10", "xa90_10", "rmin3", "rmin10", "real_ast10",
      "xa_vs_odds3", "xa_vs_odds10", "xa_strong10", "xa_weak10", "xa_trend"]   # xg* for score1
XG = ["rating_xg_for", "rating_xg_against"]      # attack/defence split expected goals
SETS = {
    "A": BASE,
    "B": BASE + FIX,
    "C": BASE + FIX + WIN,
    "D": BASE + FIX + WIN + XA,
    "E": BASE + FIX + WIN + XG,          # serving-feasible (no on-pitch data needed)
    "F": BASE + FIX + WIN + XA + XG,     # everything
    "G": BASE + ["rating_win", "rating_opp"] + XG,   # the deployed DEFENSIVE feature set (pure GBM)
}


# ----------------------------------------------------------------------------------------- data
def add_xa_features(pairs, archive, real, stat_col, stat="assist"):
    """Per (season, FPL id, gameweek) on-pitch frame joined to that week's odds, then trailing.
    Feature names say xa*, but for score1 the on-pitch quantity is xG and the real outcome is a goal."""
    real = real.copy()
    if stat == "score1":
        real["xa"] = real["xg"]
    real["real_assist"] = real[f"real_{stat}"] if f"real_{stat}" in real.columns else real["real_assist"]
    a = archive[["Season", "player_id", "Gameweek", stat_col, "F1 Win", "F1 Opponent Win"]].copy()
    a["player_id"] = pd.to_numeric(a["player_id"], errors="coerce")
    a["Gameweek"] = pd.to_numeric(a["Gameweek"], errors="coerce")
    a = a.dropna(subset=["player_id", "Gameweek"]).rename(columns={"player_id": "id"})
    r = real[["Season", "id", "Gameweek", "minutes", "xa", "real_assist"]].merge(
        a, on=["Season", "id", "Gameweek"], how="left")
    r["odds"] = pd.to_numeric(r[stat_col], errors="coerce")
    r["xa_vs_odds"] = (r["xa"] - r["odds"]).where(r["odds"].notna())   # over/under-performing his price
    easier = pd.to_numeric(r["F1 Win"], errors="coerce") >= pd.to_numeric(r["F1 Opponent Win"], errors="coerce")
    r["xa_weak"] = r["xa"].where(easier)          # xA against sides we were favoured over
    r["xa_strong"] = r["xa"].where(~easier & r["F1 Win"].notna())
    r["xa90"] = (r["xa"] / r["minutes"].where(r["minutes"] > 0) * 90).where(r["minutes"] >= 30)

    pairs = pairs.copy()
    pid = pd.to_numeric(pairs["player_id"], errors="coerce").astype(float)
    idx = list(zip(pairs["season"], pid, pairs["gw_from"]))

    def trail(col, w):
        return _dense_trailing(r, col, (w,), key="id")

    for w in (3, 6, 10):
        t = trail("xa", w)
        pairs[f"xa{w}"] = t[f"mean{w}"].reindex(idx).values
        if w == 10:
            pairs["xa_n10"] = t[f"count{w}"].reindex(idx).values
    for w in (3, 10):
        pairs[f"rmin{w}"] = trail("minutes", w)[f"mean{w}"].reindex(idx).values
        pairs[f"xa_vs_odds{w}"] = trail("xa_vs_odds", w)[f"mean{w}"].reindex(idx).values
    pairs["xa90_10"] = trail("xa90", 10)["mean10"].reindex(idx).values
    pairs["real_ast10"] = trail("real_assist", 10)["mean10"].reindex(idx).values
    pairs["xa_strong10"] = trail("xa_strong", 10)["mean10"].reindex(idx).values
    pairs["xa_weak10"] = trail("xa_weak", 10)["mean10"].reindex(idx).values
    pairs["xa_trend"] = pairs["xa3"] - pairs["xa10"]
    return pairs


def build(stat):
    archive = load_archive()
    pairs = build_pairs(archive, win_fn=rating_win_fn(archive))      # rating-fed win/opp + expected goals
    pairs = pairs[pairs["stat"] == stat].reset_index(drop=True)
    pairs["rating_win"], pairs["rating_opp"] = pairs["win_hat"], pairs["opp_hat"]
    pairs = add_form(pairs, archive, windows=(3, 6, 10))
    pairs = add_diffs(pairs)
    real = load_real_gw()
    assert real["minutes"].mean() < 100, "By-Gameweek dump looks cumulative, not per-GW"
    pairs = add_xa_features(pairs, archive, real, STATS[stat], stat)
    pairs = pairs[pairs["valid_opp"]].dropna(subset=["actual", "persistence"]).reset_index(drop=True)
    pairs["t"] = pairs["season"].astype(str) + ":" + pairs["gw_from"].astype(int).astype(str).str.zfill(2)
    pairs["t_to"] = pairs["season"].astype(str) + ":" + pairs["gw_to"].astype(int).astype(str).str.zfill(2)
    return pairs


# --------------------------------------------------------------------------------------- folds
def make_folds(df):
    """Walk forward over prediction weeks T (season-ordered). Train = outcomes resolved before T,
    val = outcomes resolved AT T, test = forecasts made at T. Growing train."""
    out = []
    for T in sorted(df["t"].unique()):
        tr, va, te = (df["t_to"] < T), (df["t_to"] == T), (df["t"] == T)
        if tr.sum() >= MIN_TRAIN and va.sum() >= 50 and te.sum():
            out.append((T, tr.values, va.values, te.values))
    return out


def prep(df, feats):
    X = df[feats].copy()
    for c in feats:
        X[c] = X[c].astype("category") if c in CATEGORICAL else pd.to_numeric(X[c], errors="coerce")
    return X


def fit_es(params, Xtr, ytr, Xva, yva):
    m = lgb.LGBMRegressor(**params)
    m.fit(Xtr, ytr, eval_X=Xva, eval_y=yva, eval_metric="l1",
          callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    return m


def predict(m, X, n=None):
    return np.clip(m.predict(X, num_iteration=n), 0, 1)


def mae(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.abs(a[ok] - b[ok]).mean()) if ok.any() else float("nan")


def run_set(df, feats, params, folds, refit=True):
    """Per fold: early-stop on VAL to get the tree count, then refit train+val with that count."""
    X, y = prep(df, feats), df["actual"].astype(float).values
    rows, iters = [], []
    for T, tr, va, te in folds:
        m = fit_es(params, X[tr], y[tr], X[va], y[va])
        n_iter = int(m.best_iteration_ or params["n_estimators"])
        iters.append(n_iter)
        val_pred = predict(m, X[va], n_iter)
        if refit:
            m2 = lgb.LGBMRegressor(**{**params, "n_estimators": n_iter})
            m2.fit(X[tr | va], y[tr | va])
            te_pred = predict(m2, X[te])
        else:
            te_pred = predict(m, X[te], n_iter)
        rows.append(df[te].assign(model=te_pred, fold=T, role="test"))
        rows.append(df[va].assign(model=val_pred, fold=T, role="val"))
    return pd.concat(rows, ignore_index=True), iters


def pooled_val(df, feats, params, folds):
    X, y = prep(df, feats), df["actual"].astype(float).values
    p, a = [], []
    for T, tr, va, te in folds:
        m = fit_es(params, X[tr], y[tr], X[va], y[va])
        p.append(predict(m, X[va], m.best_iteration_))
        a.append(y[va])
    return mae(np.concatenate(p), np.concatenate(a))


# ------------------------------------------------------------------------------------- scoring
def score(out, w_blend, label):
    """Like-for-like: only rows where the baseline exists. Returns per-horizon + pooled table."""
    t = out[(out["role"] == "test") & out["predicted"].notna()].copy()
    pers = t["persistence"].astype(float)
    t["blB"] = w_blend * t["predicted"] + (1 - w_blend) * pers
    t["blM"] = w_blend * t["model"] + (1 - w_blend) * pers
    # stack weight fitted on VAL rows only (never on test)
    v = out[(out["role"] == "val") & out["predicted"].notna()]
    vb = w_blend * v["predicted"] + (1 - w_blend) * v["persistence"].astype(float)
    grid = np.linspace(0, 1, 21)
    wstack = grid[int(np.argmin([mae(g * v["model"] + (1 - g) * vb, v["actual"]) for g in grid]))]
    t["stack"] = wstack * t["model"] + (1 - wstack) * t["blB"]
    lines = []
    for k in sorted(t["horizon"].unique()):
        s = t[t["horizon"] == k]
        lines.append((f"F{int(k)+1}", len(s), mae(s["persistence"], s["actual"]), mae(s["blB"], s["actual"]),
                      mae(s["model"], s["actual"]), mae(s["blM"], s["actual"]), mae(s["stack"], s["actual"])))
    lines.append(("all", len(t), mae(t["persistence"], t["actual"]), mae(t["blB"], t["actual"]),
                  mae(t["model"], t["actual"]), mae(t["blM"], t["actual"]), mae(t["stack"], t["actual"])))
    # SKILL vs the positional-mean forecast (the optimiser's reliability-weight yardstick):
    # skill = 1 - MAE(x) / MAE(position mean), per horizon, for each candidate serving path
    if "position_mean" in t.columns:
        print(f"\n[{label}] skill vs positional mean, per horizon  (1 - MAE/MAE_posmean)")
        print(f"{'horizon':<8}{'n':>6} | {'posmean':>8} | {'blend':>7}{'model':>7}{'blModel':>8}{'stack':>7}")
        for k in sorted(t["horizon"].unique()):
            s = t[t["horizon"] == k]
            pm = mae(s["position_mean"], s["actual"])
            sk = lambda c: 1 - mae(s[c], s["actual"]) / pm  # noqa: E731
            print(f"F{int(k)+1:<7}{len(s):>6} | {pm:>8.4f} | {sk('blB'):>7.3f}{sk('model'):>7.3f}{sk('blM'):>8.3f}{sk('stack'):>7.3f}")
    for T in sorted(t["fold"].unique()):                     # per prediction week (cross-season check)
        s = t[t["fold"] == T]
        lines.append((f"@{T[-5:]}", len(s), mae(s["persistence"], s["actual"]), mae(s["blB"], s["actual"]),
                      mae(s["model"], s["actual"]), mae(s["blM"], s["actual"]), mae(s["stack"], s["actual"])))
    print(f"\n[{label}]  stack weight on model (from VAL) = {wstack:.2f}")
    print(f"{'horizon':<8}{'n':>6} | {'persist':>8}{'blend':>8} | {'model':>8}{'blModel':>9}{'stack':>8} | {'blM vs blend':>13}{'stack vs blend':>15}")
    for h, n, p, b, m, bm, st in lines:
        print(f"{h:<8}{n:>6} | {p:>8.4f}{b:>8.4f} | {m:>8.4f}{bm:>9.4f}{st:>8.4f} | "
              f"{(b-bm)/b*100:>+12.1f}%{(b-st)/b*100:>+14.1f}%")
    return t


def bootstrap(t, col_a, col_b, n=3000, seed=0):
    """Paired bootstrap over players of MAE(col_a) - MAE(col_b) (positive = b better)."""
    rng = np.random.default_rng(seed)
    d = (t[col_a] - t["actual"]).abs() - (t[col_b] - t["actual"]).abs()
    byp = d.groupby(t["player"]).agg(["sum", "size"])
    boots = np.array([byp["sum"].values[i].sum() / byp["size"].values[i].sum()
                      for i in (rng.integers(0, len(byp), len(byp)) for _ in range(n))])
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), (boots > 0).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stat", default="assist")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--sets", default="A,B,C,D,E,F")
    ap.add_argument("--tune-on", default="F", help="feature set used to pick the common hyperparameters")
    ap.add_argument("--params", default=None,
                    help="JSON dict of hyperparameters to use instead of tuning (reuse a previous run's best)")
    args = ap.parse_args()
    w_blend = config.PROJECTION_BLEND.get(args.stat, 1.0)

    df = build(args.stat)
    folds = make_folds(df)
    print(f"{args.stat}: {len(df)} rows, {df['predicted'].notna().sum()} with baseline, "
          f"{len(folds)} folds (T = {folds[0][0]} .. {folds[-1][0]})")
    print("fold  train   val  test")
    for T, tr, va, te in folds:
        print(f"{T:<12}{tr.sum():>6}{va.sum():>6}{te.sum():>6}")

    if args.params:
        import json
        best = {**FIXED, **json.loads(args.params)}
        best["num_leaves"] = min(2 ** best["max_depth"] - 1, 255)
        print("\nusing given hyperparameters:", {k: best[k] for k in SPACE})
    else:
        rng = random.Random(SEED)
        cands = [_sample_params(rng) for _ in range(args.trials)]
        tune_feats = SETS[args.tune_on]
        print(f"\ntuning {args.trials} candidates on set {args.tune_on} by pooled VAL MAE ...")
        scores = [pooled_val(df, tune_feats, p, folds) for p in cands]
        best = cands[int(np.argmin(scores))]
        print("best:", {k: best[k] for k in SPACE}, f"pooled VAL MAE {min(scores):.4f}")

    results = {}
    for name in args.sets.split(","):
        out, iters = run_set(df, SETS[name], best, folds)
        print(f"\n=== set {name} ({len(SETS[name])} features)  early-stopped trees per fold: {iters}")
        results[name] = score(out, w_blend, f"set {name}")
        # feature importance from a final fit on everything (gain)
        X = prep(df, SETS[name])
        m = lgb.LGBMRegressor(**{**best, "n_estimators": int(np.median(iters))}).fit(X, df["actual"].astype(float))
        imp = pd.Series(m.booster_.feature_importance("gain"), index=SETS[name]).sort_values(ascending=False)
        imp = (imp / imp.sum() * 100).round(1)
        print("top features (% gain):", ", ".join(f"{k} {v}" for k, v in imp.head(10).items()))

    print("\n=== paired bootstrap over players, pooled TEST, like-for-like rows ===")
    for name, t in results.items():
        for col, lab in (("blM", "blended model"), ("stack", "stack")):
            d, lo, hi, p = bootstrap(t, "blB", col)
            print(f"set {name} {lab:<14}: gain vs blend {d:+.5f}  95% [{lo:+.5f}, {hi:+.5f}]  P(better) = {p:.2f}")


if __name__ == "__main__":
    main()
