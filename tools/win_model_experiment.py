"""Can we beat the incumbent F3+ win-probability model?

    python tools/win_model_experiment.py

Fits candidate models on outputs/team_gw_real_odds_long.csv (built by
tools/team_gw_odds_table.py) and compares them against the coefficients currently
serving F3-F8, on identical folds.

METHODOLOGY NOTES — both matter more than the model choice here:

1. GROUPED SPLITS, NEVER RANDOM. One match appears in up to 8 rows, once per horizon it
   was forecast from, all sharing the same `Win` value. A random split puts the same
   match on both sides and reports a fantasy score. Folds are grouped by target
   gameweek, so a match and all its horizons stay together.

2. TWO DATASETS. Top 6 odds only exist from GW21, so requiring them costs 3.3x the rows
   and five gameweeks of variation (200 rows from 4 gameweeks vs 660 from 9). Both are
   evaluated; the no-Top-6 set is the one with enough independent variation to trust.

Feature ideas under test: raw outrights, pairwise DIFFERENCES between them (the market's
view of the mismatch, which is what a match price really encodes), log-odds transforms
(relegation odds span 0.0005-0.5, so levels are wildly skewed), and a HORIZON term — the
incumbent applies one formula to every fixture from F3 to F8, but distant fixtures should
regress toward the mean and may need their own slope.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config, model  # noqa: E402

LONG = os.path.join(config.OUTPUTS_DIR, "team_gw_real_odds_long.csv")
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def features(d, use_top6):
    """Return {name: DataFrame} of candidate designs."""
    home = (d["Venue"] == "H").astype(float)
    t, r = d["Title"], d["Relegation"]
    ot, orl = d["Opponent Title"], d["Opponent Relegation"]

    raw = {"title": t, "releg": r, "opp_title": ot, "opp_releg": orl, "home": home}
    if use_top6:
        raw |= {"top6": d["Top 6"], "opp_top6": d["Opponent Top 6"]}

    # Differences: a match price encodes the MISMATCH, not the absolute levels
    diff = dict(raw)
    diff |= {"title_diff": t - ot, "releg_diff": r - orl,
             "strength_diff": (t - r) - (ot - orl)}
    if use_top6:
        diff["top6_diff"] = d["Top 6"] - d["Opponent Top 6"]

    # Log-odds: relegation spans ~0.0005-0.5, so levels are hopelessly skewed
    lg = {"lt": logit(t), "lr": logit(r), "lot": logit(ot), "lor": logit(orl),
          "home": home, "lt_diff": logit(t) - logit(ot), "lr_diff": logit(r) - logit(orl)}
    if use_top6:
        lg |= {"l6": logit(d["Top 6"]), "lo6": logit(d["Opponent Top 6"]),
               "l6_diff": logit(d["Top 6"]) - logit(d["Opponent Top 6"])}

    horizon = dict(diff, fixture=d["Fixture"].astype(float))
    lg_h = dict(lg, fixture=d["Fixture"].astype(float))
    # does the mismatch signal weaken with distance?
    lg_hx = dict(lg_h, fix_x_diff=d["Fixture"].astype(float) * (logit(t) - logit(ot)))

    return {"raw levels": raw, "+ diffs": diff, "+ diffs + horizon": horizon,
            "log-odds + diffs": lg, "log-odds + horizon": lg_h,
            "log-odds + horizon x diff": lg_hx}


def cv_mae(X, y, groups, n_splits=5):
    X = pd.DataFrame(X).astype(float)
    errs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = RidgeCV(alphas=np.logspace(-4, 2, 25)).fit(X.iloc[tr], y.iloc[tr])
        errs.append(np.abs(m.predict(X.iloc[te]) - y.iloc[te]).mean())
    return float(np.mean(errs))


def incumbent_mae(d, groups, n_splits=5):
    """The live model, scored on the same folds' test halves (it is already fitted, so
    every fold's test set is simply held out)."""
    pred = model.win_pred(d["Title"], d["Relegation"], d["Top 6"],
                          d["Opponent Title"], d["Opponent Relegation"],
                          d["Opponent Top 6"], d["Venue"].eq("H"))
    ok = pred.notna()
    errs = []
    for _, te in GroupKFold(n_splits=n_splits).split(d, d["Win"], groups):
        sel = d.index[te][ok.iloc[te].values]
        if len(sel):
            errs.append((pred.loc[sel] - d.loc[sel, "Win"]).abs().mean())
    return float(np.mean(errs))


def run(d, label, use_top6):
    d = d.reset_index(drop=True)
    y = d["Win"]
    groups = d["Season"].astype(str) + "-" + d["Target Gameweek"].astype(str)
    n_groups = groups.nunique()
    splits = min(5, n_groups)

    print(f"\n{'=' * 72}\n{label}: {len(d)} rows, {n_groups} target gameweeks, "
          f"{splits}-fold grouped CV\n{'=' * 72}")

    baseline = np.abs(y - y.mean()).mean()
    print(f"{'predict the mean':<30}{baseline:>9.4f}")
    if use_top6:
        print(f"{'incumbent win_pred_f3plus':<30}{incumbent_mae(d, groups, splits):>9.4f}")

    results = {}
    for name, feats in features(d, use_top6).items():
        results[name] = cv_mae(feats, y, groups, splits)
        print(f"{name:<30}{results[name]:>9.4f}")
    return results


if __name__ == "__main__":
    model.load_coefficients()
    long = pd.read_csv(LONG)
    top6_cols = ["Top 6", "Opponent Top 6"]
    other = [c for c in long.columns if c not in top6_cols]

    run(long.dropna(subset=other + top6_cols), "WITH Top 6 (complete rows)", True)
    run(long.dropna(subset=other), "WITHOUT Top 6 (3.3x the rows)", False)
