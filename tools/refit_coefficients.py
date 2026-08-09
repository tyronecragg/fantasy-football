"""Refit every model coefficient from the historical archives — replacing the old
workflow of pasting data into modelling scripts and hand-copying coefficients back.

Usage:
    python tools/refit_coefficients.py            # dry run: fit, compare, write nothing
    python tools/refit_coefficients.py --write    # regenerate fpl_pipeline/data/coefficients.json
                                                  # (workbook original backed up once to
                                                  #  coefficients_workbook.json for parity mode)

Design notes:
- Feature construction reuses fpl_pipeline.model._features / win_pred's feature map, so
  training and serving can never drift apart.
- Each model keeps its existing functional form (same feature set as the current
  coefficients.json); fitting is plain OLS (np.linalg.lstsq).
- Season-odds probabilities are computed with the serving margin (1.08), fixing the
  workbook's train/serve skew (its Historical sheet used 1.03 for everything).
- The bonus model is carried over unchanged: refitting it needs actual bonus-point
  outcomes, which the archives don't contain.
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config, model  # noqa: E402
from fpl_pipeline.markets import implied  # noqa: E402

PLAYER_HISTORY = os.path.join(config.INPUTS_DIR, "historical_player_data.csv")
FIXTURE_HISTORY = os.path.join(config.INPUTS_DIR, "historical_fixture_odds.csv")

BASELINE_TARGETS = {
    "score1": "F1 Score 1+", "assist": "F1 Assist", "yellow": "F1 Yellow Card",
    "concede2": "F1 Concede 2+ Goals", "concede4": "F1 Concede 4+ Goals",
    "saves3": "F1 3+ Saves", "saves6": "F1 6+ Saves",
    "clean_sheet": "F1 Clean Sheet", "pred_xp": "F1 XP",
}
MIN_ROWS = 100


def ols(feature_dict, feature_names, target):
    """Least-squares fit over named features; returns (coefs dict, n, r2) or None."""
    y = np.asarray(target, dtype=float)
    cols = []
    for k in feature_names:
        v = np.asarray(feature_dict[k], dtype=float)
        cols.append(np.full(len(y), float(v)) if v.ndim == 0 else v)
    X = np.column_stack(cols)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if mask.sum() < MIN_ROWS:
        return None
    X, y = X[mask], y[mask]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1.0 - resid.var() / y.var() if y.var() > 0 else float("nan")
    return dict(zip(feature_names, (round(float(b), 6) for b in beta))), int(mask.sum()), r2


def fit_baselines(report, path=None):
    hist = pd.read_csv(path or PLAYER_HISTORY)
    hist = hist.loc[:, ~hist.columns.str.startswith("Unnamed")]
    for col in hist.columns:
        if col not in ("Season", "Player Name", "Position", "Team", "F1 Opponent", "F1 Venue"):
            hist[col] = pd.to_numeric(hist[col], errors="coerce")

    feats = model._features(hist["F1 Win"], hist["F1 Opponent Win"],
                            hist["Position"], hist["F1 Venue"] == "H")

    out = {}
    for stat, target_col in BASELINE_TARGETS.items():
        names = list(model.BASELINES[stat].keys())  # keep the model's functional form
        fit = ols(feats, names, hist[target_col])
        if fit is None:
            report.append((f"baseline:{stat}", "SKIPPED (too few rows) — kept existing", None, None))
            out[stat] = model.BASELINES[stat]
        else:
            coefs, n, r2 = fit
            report.append((f"baseline:{stat}", "refit", n, r2))
            out[stat] = coefs
    return out


def _fixture_perspectives():
    """Each historical fixture yields two training rows (home and away perspective)."""
    hist = pd.read_csv(FIXTURE_HISTORY)
    for col in hist.columns:
        if col not in ("Season", "home_team", "away_team"):
            hist[col] = pd.to_numeric(hist[col], errors="coerce")

    def side(prefix, opp_prefix, home):
        return pd.DataFrame({
            "win": implied(hist[f"{prefix}_win_odds"], config.MARGIN_WDW),
            "title": implied(hist[f"{prefix}_title_odds"], config.MARGIN_SEASON),
            "releg": implied(hist[f"{prefix}_relegation_odds"], config.MARGIN_SEASON),
            "top6": implied(hist[f"{prefix}_top_6_odds"], config.MARGIN_SEASON),
            "opp_title": implied(hist[f"{opp_prefix}_title_odds"], config.MARGIN_SEASON),
            "opp_releg": implied(hist[f"{opp_prefix}_relegation_odds"], config.MARGIN_SEASON),
            "opp_top6": implied(hist[f"{opp_prefix}_top_6_odds"], config.MARGIN_SEASON),
            "home": float(home),
        })

    return pd.concat([side("home", "away", 1), side("away", "home", 0)], ignore_index=True)


def fit_win_models(report):
    rows = _fixture_perspectives()
    sd = (rows["title"] + rows["top6"] - rows["releg"]) - \
         (rows["opp_title"] + rows["opp_top6"] - rows["opp_releg"])

    win_feats = {
        "const": np.ones(len(rows)), "strength_diff": sd, "home": rows["home"],
        "top6_share": rows["top6"] / (rows["top6"] + rows["opp_top6"] + 0.01),
        "title_diff": rows["title"] - rows["opp_title"],
        "home_x_strength_diff": rows["home"] * sd,
        "opp_top6": rows["opp_top6"], "title": rows["title"],
        "strength_diff_sq": sd ** 2, "abs_strength_diff": sd.abs(),
    }
    win_fit = ols(win_feats, list(model.COEFS["win_pred_f3plus"].keys()), rows["win"])

    opp_feats = {
        "const": np.ones(len(rows)), "title": rows["title"], "releg": rows["releg"],
        "opp_title": rows["opp_title"], "opp_releg": rows["opp_releg"], "home": rows["home"],
    }
    opp_fit = ols(opp_feats, ["const", "title", "releg", "opp_title", "opp_releg", "home"],
                  rows["win"])

    win_pred = model.COEFS["win_pred_f3plus"]
    if win_fit:
        win_pred, n, r2 = win_fit
        report.append(("win_pred_f3plus", "refit", n, r2))
    else:
        report.append(("win_pred_f3plus", "SKIPPED — kept existing", None, None))

    match_odds = {k: v for k, v in model.SHEET.items() if k.startswith("Match Odds")}
    if opp_fit:
        coefs, n, r2 = opp_fit
        match_odds = {
            "Match Odds Intercept": coefs["const"], "Match Odds Winner": coefs["title"],
            "Match Odds Relegation": coefs["releg"], "Match Odds Opponent Winner": coefs["opp_title"],
            "Match Odds Opponent Relegation": coefs["opp_releg"], "Match Odds Home": coefs["home"],
        }
        report.append(("opp_win_pred (Match Odds)", "refit", n, r2))
    else:
        report.append(("opp_win_pred (Match Odds)", "SKIPPED — kept existing", None, None))
    return win_pred, match_odds


def fit_f2_score_model(report):
    """The Coefficients-sheet score model (rows 1-5): P(score 1+) ~ win, opp, diff, venue."""
    hist = pd.read_csv(PLAYER_HISTORY)
    win = pd.to_numeric(hist["F1 Win"], errors="coerce")
    opp = pd.to_numeric(hist["F1 Opponent Win"], errors="coerce")
    feats = {
        "const": np.ones(len(hist)), "win": win, "opp": opp, "diff": win - opp,
        "home": (hist["F1 Venue"] == "H").astype(float),
    }
    fit = ols(feats, ["const", "win", "opp", "diff", "home"],
              pd.to_numeric(hist["F1 Score 1+"], errors="coerce"))
    score = {k: v for k, v in model.SHEET.items() if k.startswith("Score")}
    if fit:
        coefs, n, r2 = fit
        score = {"Score Intercept": coefs["const"], "Score Difficulty": coefs["win"],
                 "Score Opponent": coefs["opp"], "Score Diff": coefs["diff"],
                 "Score Venue": coefs["home"]}
        report.append(("f2_score (Score rows 1-5)", "refit", n, r2))
    else:
        report.append(("f2_score (Score rows 1-5)", "SKIPPED — kept existing", None, None))
    return score


def _load_backtest():
    import importlib.util
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_projections.py")
    spec = importlib.util.spec_from_file_location("backtest_projections", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("backtest_projections", mod)
    spec.loader.exec_module(mod)
    return mod


def projection_holdout_check(holdout_gws=4, tolerance=0.05):
    """Guardrail before --write: candidate baselines must not degrade the F2-F6
    projections. The baselines are consumed as a RATIO (factor = odds / baseline at M,
    prediction = factor x baseline at N), so a refit that improves same-week fit can
    still wreck projections when its surface misbehaves between contexts — observed
    2025-26: baselines refit on 7 gameweeks degraded projection MAE by 100-180%.

    Fits on all but the last `holdout_gws` archived gameweeks, then compares
    forecast-vs-actual MAE on pairs originating in the holdout, current vs candidate.
    Returns True only if no stat degrades by more than `tolerance` relative MAE.
    """
    import tempfile

    bt = _load_backtest()
    archive = bt.load_archive()
    gws = sorted(archive["Gameweek"].dropna().unique())
    if len(gws) < holdout_gws + 4:
        print(f"Holdout check SKIPPED: only {len(gws)} archived gameweeks "
              f"(need {holdout_gws + 4}) — refusing is the safe default")
        return False
    cutoff = gws[-holdout_gws]

    train_path = os.path.join(tempfile.gettempdir(), "refit_train_history.csv")
    archive[archive["Gameweek"] < cutoff].to_csv(train_path, index=False)
    candidate = fit_baselines([], path=train_path)

    def holdout_mae(pairs):
        sub = pairs[pairs["valid_opp"] & (pairs["gw_from"] >= cutoff)]
        return {stat: (s["predicted"] - s["actual"]).abs().mean()
                for stat, s in sub.dropna(subset=["predicted", "actual"]).groupby("stat")}

    current = holdout_mae(bt.build_pairs(archive))
    model.BASELINES.update(candidate)
    try:
        refit_mae = holdout_mae(bt.build_pairs(archive))
    finally:
        model.load_coefficients()

    print(f"\nHoldout projection check (fit < GW{int(cutoff)}, evaluate >= GW{int(cutoff)}):")
    degraded = []
    for stat in current:
        change = (refit_mae[stat] - current[stat]) / current[stat]
        flag = "DEGRADED" if change > tolerance else "ok"
        print(f"  {stat:<12} current {current[stat]:.4f}  refit {refit_mae[stat]:.4f}  {change:+.1%}  {flag}")
        if change > tolerance:
            degraded.append(stat)
    if degraded:
        print(f"Refit would degrade projections for: {', '.join(degraded)}")
    return not degraded


def main(write=False, force=False):
    report = []
    baselines = fit_baselines(report)
    win_pred, match_odds, = fit_win_models(report)
    score_model = fit_f2_score_model(report)

    sheet = dict(model.SHEET)
    sheet.update(match_odds)
    sheet.update(score_model)

    new = {
        "_source": "Refit from inputs/historical_*.csv by tools/refit_coefficients.py",
        "baselines": baselines,
        "win_pred_f3plus": win_pred,
        "bonus": model.COEFS["bonus"],  # not refittable from the archives
        "coefficients_sheet": sheet,
        "total_xp_weights": model.COEFS["total_xp_weights"],
    }

    print(f"{'model':<28} {'status':<34} {'n':>6} {'R2':>7}")
    for name, status, n, r2 in report:
        print(f"{name:<28} {status:<34} {n if n else '':>6} {f'{r2:.3f}' if r2 is not None else '':>7}")

    changed = sum(abs(a - b) > 1e-9
                  for stat in baselines
                  for a, b in zip(baselines[stat].values(), model.BASELINES[stat].values()))
    print(f"\n{changed} baseline coefficients changed vs current file")

    if write:
        if not projection_holdout_check() and not force:
            print("\nNOT WRITTEN: the holdout check failed. Rerun with --force to override "
                  "(you probably shouldn't).")
            return new
        if not os.path.exists(model.WORKBOOK_COEFFICIENTS_JSON):
            shutil.copy(config.COEFFICIENTS_JSON, model.WORKBOOK_COEFFICIENTS_JSON)
            print(f"Backed up workbook coefficients -> {model.WORKBOOK_COEFFICIENTS_JSON}")
        with open(config.COEFFICIENTS_JSON, "w", encoding="utf-8") as fh:
            json.dump(new, fh, indent=1)
        print(f"Wrote {config.COEFFICIENTS_JSON}")
    else:
        print("Dry run — rerun with --write to update coefficients.json "
              "(a holdout projection check gates the write)")
    return new


if __name__ == "__main__":
    main(write="--write" in sys.argv, force="--force" in sys.argv)
