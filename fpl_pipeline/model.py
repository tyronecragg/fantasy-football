"""The statistical model extracted from the workbook: per-stat regression baselines,
player factors, win predictions, probability ladders, the FPL scoring rules and the
bonus model. All coefficients come from data/coefficients.json (never hardcoded here).
"""
import json
import os
from math import erf, sqrt

import numpy as np
import pandas as pd

from . import config

COEFS = None
BASELINES = None
SHEET = None

WORKBOOK_COEFFICIENTS_JSON = os.path.join(config.DATA_DIR, "coefficients_workbook.json")


def load_coefficients(path=None):
    """(Re)load the coefficient set used by every model function. Default is the live
    coefficients.json; parity validation loads coefficients_workbook.json when a refit
    has replaced the live file (tools/refit_coefficients.py --write backs it up there)."""
    global COEFS, BASELINES, SHEET
    with open(path or config.COEFFICIENTS_JSON, encoding="utf-8") as fh:
        COEFS = json.load(fh)
    BASELINES = COEFS["baselines"]
    SHEET = COEFS["coefficients_sheet"]


load_coefficients()


def _features(win, opp, pos, home):
    """Feature dict shared by every baseline regression."""
    diff = win - opp
    d = {
        "const": 1.0,
        "win": win, "opp": opp,
        "def": (pos == "DEF").astype(float), "mid": (pos == "MID").astype(float),
        "fwd": (pos == "FWD").astype(float),
        "home": home.astype(float),
        "diff": diff, "absdiff": diff.abs(), "diff2": diff ** 2,
        "win_opp": win * opp,
    }
    d["win_home"] = d["win"] * d["home"]
    d["opp_home"] = d["opp"] * d["home"]
    d["diff_home"] = d["diff"] * d["home"]
    for p in ("def", "mid", "fwd"):
        d[f"{p}_diff"] = d[p] * d["diff"]
        d[f"{p}_home"] = d[p] * d["home"]
    return d


def baseline(stat, win, opp, pos, home):
    """Predicted probability of `stat` given win/opponent-win probabilities, position and
    venue. NaN inputs propagate (like Excel errors)."""
    feats = _features(win, opp, pos, home)
    total = pd.Series(0.0, index=win.index)
    for name, coef in BASELINES[stat].items():
        total = total + coef * feats[name]
    return total


def win_pred(title, releg, top6, opp_title, opp_releg, opp_top6, home):
    """F3+ own-team win probability (the Lasso pasted into the Players sheet)."""
    c = COEFS["win_pred_f3plus"]
    sd = (title + top6 - releg) - (opp_title + opp_top6 - opp_releg)
    feats = {
        "const": 1.0,
        "strength_diff": sd,
        "home": home.astype(float),
        "top6_share": top6 / (top6 + opp_top6 + 0.01),
        "title_diff": title - opp_title,
        "home_x_strength_diff": home.astype(float) * sd,
        "opp_top6": opp_top6,
        "title": title,
        "strength_diff_sq": sd ** 2,
        "abs_strength_diff": sd.abs(),
    }
    total = pd.Series(0.0, index=title.index)
    for name, coef in c.items():
        total = total + coef * feats[name]
    return total


def opp_win_pred(own_title, own_releg, opp_title, opp_releg, home):
    """F3+ opponent win probability from the Coefficients sheet 'Match Odds' block.
    The home flag is inverted (1 when the opponent is at home). max(0, ...) with NaN
    propagating, as Excel's MAX does with errors."""
    raw = (SHEET["Match Odds Intercept"]
           + SHEET["Match Odds Winner"] * opp_title
           + SHEET["Match Odds Relegation"] * opp_releg
           + SHEET["Match Odds Opponent Winner"] * own_title
           + SHEET["Match Odds Opponent Relegation"] * own_releg
           + SHEET["Match Odds Home"] * (~home).astype(float))
    return raw.clip(lower=0)


def f2_score1(factor, win, opp, home):
    """F2 Score 1+ = Score-1+ factor x Coefficients-sheet score model (rows 1-5)."""
    return factor * (SHEET["Score Intercept"]
                     + SHEET["Score Difficulty"] * win
                     + SHEET["Score Opponent"] * opp
                     + SHEET["Score Diff"] * (win - opp)
                     + SHEET["Score Venue"] * home.astype(float))


def ladder_score2(p):
    """Step function mapping P(score 1+) -> P(score 2+) for modelled fixtures."""
    conditions = [p < 0.3, p == 0.3, p < 0.5, p < 0.55, p < 0.57, p < 0.58, p < 0.59, p < 0.6]
    values = [0.01, 0.03, 0.05, 0.06, 0.1, 0.11, 0.12, 0.15]
    out = pd.Series(np.select(conditions, values, default=0.28), index=p.index)
    return out.where(p.notna())


def ladder_score3(p):
    """Step function mapping P(score 1+) -> P(score 3+)."""
    conditions = [p < 0.3, p < 0.55, p < 0.59, p < 0.6]
    values = [0.0, 0.01, 0.02, 0.03]
    out = pd.Series(np.select(conditions, values, default=0.11), index=p.index)
    return out.where(p.notna())


def _poisson_lambda(p1):
    """Goal rate implied by P(score 1+) under a Poisson goals model: 1-exp(-lam)=p1."""
    return -np.log(1.0 - p1.clip(0.0, 0.999999))


def poisson_score2(p1):
    """Smooth P(score 2+) from P(score 1+), Poisson-consistent (improved mode's
    replacement for the ladder_score2 step function)."""
    lam = _poisson_lambda(p1)
    return (1.0 - np.exp(-lam) * (1.0 + lam)).where(p1.notna())


def poisson_score3(p1):
    """Smooth P(score 3+) from P(score 1+), Poisson-consistent."""
    lam = _poisson_lambda(p1)
    return (1.0 - np.exp(-lam) * (1.0 + lam + lam ** 2 / 2.0)).where(p1.notna())


def scale_win_pair(win, opp):
    """Force predicted win + opponent-win <= 1 (they are fitted independently and can
    jointly exceed certainty). Pairs summing above 1 are scaled down proportionally,
    which implies a zero draw share for them — the least-intervention correction.
    Inputs are assumed already clipped to [0, 1]."""
    total = win + opp
    factor = np.where(total > 1.0, 1.0 / total, 1.0)
    return win * factor, opp * factor


def dc_probability(dc90, sd, threshold):
    """P(defensive contribution >= threshold) = 1 - NormCDF(threshold; dc90, sd)."""
    def cdf(x, mu):
        if pd.isna(mu):
            return np.nan
        return 0.5 * (1.0 + erf((x - mu) / (sd * sqrt(2.0))))
    return dc90.map(lambda mu: 1.0 - cdf(threshold, mu))


# FPL scoring: goals points per position; the tail-sum P(1+)+P(2+)+P(3+) approximates E[goals]
GOAL_POINTS = {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4}


def xp_pre(pos, start, s):
    """Expected points before bonus, exactly as the workbook's AJ2-style formula:
    every component is IFERROR(...,0) so missing stats contribute 0, the appearance 2
    is unconditional, and the whole thing scales by start probability.

    `s` maps stat name -> Series: score1..score3, assist, yellow, clean_sheet,
    concede2, concede4, saves3, saves6, dc_def, dc_mid.
    """
    def z(x):
        return x.fillna(0.0)

    goals = pd.Series([GOAL_POINTS.get(p, 0) for p in pos], index=pos.index, dtype=float)
    goal_pts = goals * (z(s["score1"]) + z(s["score2"]) + z(s["score3"]))
    common = 2.0 + goal_pts + 3.0 * z(s["assist"]) - z(s["yellow"])

    is_gk = pos == "GK"
    is_def = pos == "DEF"
    is_mid = pos == "MID"

    pts = common.copy()
    pts = pts + np.where(is_gk | is_def, 4.0 * z(s["clean_sheet"]), 0.0)
    pts = pts + np.where(is_mid, 1.0 * z(s["clean_sheet"]), 0.0)
    pts = pts - np.where(is_gk | is_def, 2.0 * z(s["concede2"]) + 2.0 * z(s["concede4"]), 0.0)
    pts = pts + np.where(is_gk, z(s["saves3"]) + z(s["saves6"]), 0.0)
    pts = pts + np.where(is_def, 2.0 * z(s["dc_def"]), 0.0)
    pts = pts + np.where(is_mid, 2.0 * z(s["dc_mid"]), 0.0)

    return start.fillna(0.0) * pts


def bonus_probability(xp):
    b = COEFS["bonus"]
    return (b["intercept"] + b["slope"] * xp).clip(0.0, 1.0)


def xp_with_bonus(xp, bonus):
    return xp + 2.0 * bonus
