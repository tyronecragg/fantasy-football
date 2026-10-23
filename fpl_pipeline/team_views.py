"""Personal team views applied to the PRICED weeks (F1, and F2 where Betway prices it).

The rating-served weeks (F3-F8) take the views through rating.apply_team_views. This module shades
the bookmaker-derived numbers for the priced weeks at config.TEAM_VIEW_MARKET_WEIGHT (0.5 = half
strength), so a view tilts Betway's prices rather than replacing them.

A nudge n (goals per game, from inputs/team_views.csv) for team t against opponent o, after the
weight w is applied (n_t = w*n, n_o = w*n_opp):
  - win/loss: log(P win / P loss) += TEAM_VIEW_LOGODDS_PER_GOAL * (n_t - n_o); the draw is kept;
  - goals the team scores  x exp(s(n_t) - s(n_o)), goals it concedes x exp(s(n_o) - s(n_t)),
    where s(n) = sign(n) * log(1 + |n| / 2 / TEAM_VIEW_AVG_GOALS)  (half to attack, half to defence);
  - clean sheet / concede 2+ / 4+ / keeper saves are re-derived from the scaled conceding rate;
  - each player's goal / 2+ goals / assist / 2+ assists price is scaled by his team's attacking change,
    so his own price still sets how he compares with his teammates.
Bookings are untouched. Each market is converted to an implied Poisson rate, scaled, and converted
back, so a probability can never leave [0, 1].
"""
import numpy as np
import pandas as pd

from . import config


def _s(n):
    n = np.asarray(n, dtype=float)
    return np.sign(n) * np.log1p(np.abs(n) / 2 / config.TEAM_VIEW_AVG_GOALS)


def _tail(lam, k):
    """P(N >= k) for N ~ Poisson(lam), vectorised."""
    lam = np.asarray(lam, dtype=float)
    term, cdf = np.exp(-lam), np.zeros_like(lam)
    for j in range(k):
        cdf = cdf + term
        term = term * lam / (j + 1)
    return 1 - cdf


def _rate_from_tail(p, k):
    """The Poisson rate whose P(N >= k) equals p (bisection; NaN where p is missing or degenerate)."""
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p) & (p > 0) & (p < 1)
    lo, hi = np.full(p.shape, 1e-6), np.full(p.shape, 20.0)
    for _ in range(50):
        mid = (lo + hi) / 2
        below = _tail(mid, k) < np.where(ok, p, 0.5)
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return np.where(ok, (lo + hi) / 2, np.nan)


def _scale_tail(p, k, factor, ref_k=None, ref_p=None):
    """Scale the rate behind P(N >= k) by `factor` and return the new probability. With ref_k/ref_p
    the rate comes from a companion line (e.g. 4+ scaled off the 2+ rate) and the market's own shape
    is kept by multiplying p by tail(rate')/tail(rate)."""
    p = pd.to_numeric(p, errors="coerce")
    factor = np.asarray(factor, dtype=float)
    if ref_k is None:
        lam = _rate_from_tail(p.values, k)
        out = _tail(lam * factor, k)
    else:
        lam = _rate_from_tail(pd.to_numeric(ref_p, errors="coerce").values, ref_k)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = p.values * _tail(lam * factor, k) / _tail(lam, k)
    out = np.where(np.isfinite(out), np.clip(out, 0, 1), p.values)   # unpriced rows pass through
    return pd.Series(out, index=p.index)


def shade(teamview, mkts, roster, views, weight=None):
    """Return (teamview, mkts, report) with the priced weeks shaded by `views` {team: goals/game}.
    Inputs are not modified. `roster` needs name/team (maps player markets to teams)."""
    weight = config.TEAM_VIEW_MARKET_WEIGHT if weight is None else weight
    if not views or weight <= 0:
        return teamview, mkts, {}
    eff = {t: n * weight for t, n in views.items()}
    tv, mk = teamview.copy(), dict(mkts)
    gfor = {}
    for wk in ("f1", "f2"):
        if f"{wk}_opponent" not in tv:
            continue
        nt = tv["team"].map(eff).fillna(0.0).values
        no = pd.Series(tv[f"{wk}_opponent"]).map(eff).fillna(0.0).values
        gfor[wk] = pd.Series(np.exp(_s(nt) - _s(no)), index=tv["team"].values)
        pw = pd.to_numeric(tv[f"{wk}_win"], errors="coerce").values
        pl = pd.to_numeric(tv[f"{wk}_opponent_win"], errors="coerce").values
        ok = np.isfinite(pw) & np.isfinite(pl) & (pw > 0) & (pl > 0)
        tot = pw + pl
        with np.errstate(invalid="ignore", divide="ignore"):
            r = pw / pl * np.exp(config.TEAM_VIEW_LOGODDS_PER_GOAL * (nt - no))
            new_w = tot * r / (1 + r)
        tv[f"{wk}_win"] = np.where(ok, new_w, pw)
        tv[f"{wk}_opponent_win"] = np.where(ok, tot - new_w, pl)

    def against(wk, teams):      # conceding-rate factor for each row's team
        g = gfor.get(wk)
        return np.ones(len(teams)) if g is None else 1 / teams.map(g).fillna(1.0).values

    for key, wk in (("clean_sheet", "f1"), ("f2_clean_sheet", "f2")):
        if key in mk and len(mk[key]):
            d = mk[key].copy()
            lam = -np.log(pd.to_numeric(d["prob"], errors="coerce").clip(1e-9, 1))
            d["prob"] = np.exp(-lam * against(wk, d["team"])).where(d["prob"].notna(), d["prob"])
            mk[key] = d
    for key, wk in (("concede", "f1"), ("f2_concede", "f2")):
        if key in mk and len(mk[key]):
            d = mk[key].copy()
            f = against(wk, d["team"])
            p2 = d["prob2"].copy()
            d["prob2"] = _scale_tail(p2, 2, f).values
            d["prob4"] = _scale_tail(d["prob4"], 4, f, ref_k=2, ref_p=p2).values
            mk[key] = d
    if "gk_saves" in mk and len(mk["gk_saves"]):
        d = mk["gk_saves"].copy()
        f = against("f1", d["team"])                     # more goals against ~ more shots to save
        p3 = d["prob3"].copy()
        d["prob3"] = _scale_tail(p3, 3, f).values
        d["prob6"] = _scale_tail(d["prob6"], 6, f, ref_k=3, ref_p=p3).values
        mk["gk_saves"] = d

    team_of = dict(zip(roster["name"], roster["team"]))
    g1 = gfor.get("f1", pd.Series(dtype=float))
    for key, k in (("score1", 1), ("score2", 2), ("assist", 1), ("assist2", 2)):
        if key in mk and len(mk[key]):
            d = mk[key].copy()
            f = d["player"].map(team_of).map(g1).fillna(1.0).values
            d["prob"] = _scale_tail(d["prob"], k, f).values
            mk[key] = d
    report = {t: round(n, 3) for t, n in eff.items()}
    return tv, mk, report
