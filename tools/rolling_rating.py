# -*- coding: utf-8 -*-
"""Rolling odds-implied team ratings: can they predict a match's closing price k weeks ahead better
than a static prior-season rating (and the incumbent outrights model)?  TEST HARNESS ONLY.

    env/Scripts/python tools/rolling_rating.py [--grid] [--seasons 2122 2223 2324 2425 2526]

IDEA. Every closing 1X2 price says "home strength - away strength + home advantage = this". With ten
matches a week, a few weeks of prices pin down every team's strength far more precisely than three
coarse outright markets. Ratings are solved by weighted least squares on the draw-neutral log-odds
  y = log(pH / pA) = r_home - r_away + hfa
over all matches played BEFORE a cutoff date, weighted by recency (half-life in days) and discounted
by season (delta per season back). Promoted sides get a pseudo-observation at the prior season's
three-lowest level (the substitution tools/build_team_ratings.py validated).

PREDICTION. For a match k weeks ahead, the cutoff is match_date - 7k days. The rating difference at
that cutoff maps to (P home, P draw, P away) through a small multinomial logistic fit per horizon on
the OTHER seasons (leave-one-season-out) against the closing probabilities (soft targets), so far
horizons regress toward even by fitted slope rather than by assumption.

SCORE. MAE of the win probability from both perspectives (home P(H) and away P(A)) vs the closing
price, per horizon and by season phase. Baselines: predict-the-mean, the static prior-season rating
(the existing tool's construction), and on 2025-26 archive matches the incumbent outrights model.
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fpl_pipeline import config, model  # noqa: E402
from fpl_pipeline.rating import solve  # noqa: E402  (the served solver - one implementation)
from tools.build_team_ratings import DEFAULT_SEASONS, fetch, match_probs, season_label  # noqa: E402

HORIZONS = range(1, 9)          # weeks ahead; F{k+1} in pipeline terms
RIDGE = 1e-3


# ----------------------------------------------------------------------------------------- data
def load_matches(seasons):
    frames = []
    for yy in seasons:
        raw = fetch(yy)
        d = match_probs(raw)
        d["date"] = pd.to_datetime(raw.loc[d.index, "Date"], dayfirst=True)
        d["season"] = season_label(yy)
        d["p_draw"] = 1 - d["p_home"] - d["p_away"]
        frames.append(d[["season", "date", "HomeTeam", "AwayTeam", "p_home", "p_draw", "p_away"]])
    m = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    m["y"] = np.log(m["p_home"] / m["p_away"])
    # round index within season (1 = opening weekend) from the date order of distinct match dates
    m["round"] = m.groupby("season")["date"].rank(method="dense").astype(int)
    m["season_idx"] = m["season"].map({s: i for i, s in enumerate(sorted(m["season"].unique()))})
    return m


def season_teams(m):
    return {s: sorted(set(g["HomeTeam"]) | set(g["AwayTeam"])) for s, g in m.groupby("season")}


def final_ratings(m, season):
    """Unweighted full-season ratings (sum-zero) - used only to set the promoted level."""
    g = m[m["season"] == season]
    teams = sorted(set(g["HomeTeam"]) | set(g["AwayTeam"]))
    r, _ = solve(g, teams, np.ones(len(g)), {})
    return r


# --------------------------------------------------------------------------------------- rating
class Rater:
    def __init__(self, m, halflife, delta, w0, static=False):
        self.m, self.halflife, self.delta, self.w0, self.static = m, halflife, delta, w0, static
        self.teams = season_teams(m)
        self.seasons = sorted(self.teams)
        self.level = {}
        for prev, cur in zip(self.seasons, self.seasons[1:]):
            fr = final_ratings(m, prev)
            self.level[cur] = float(fr.nsmallest(3).mean())
        self._cache = {}

    def ratings(self, cutoff, season):
        key = (cutoff, season)
        if key in self._cache:
            return self._cache[key]
        si = self.seasons.index(season)
        prev = self.seasons[si - 1] if si else None
        if self.static:                                   # prior season only, equal weights
            g = self.m[self.m["season"] == prev] if prev else self.m.iloc[0:0]
            w = np.ones(len(g))
        else:
            g = self.m[self.m["date"] < cutoff]
            back = si - g["season_idx"].values
            g = g[back <= 2]
            back = back[back <= 2]
            age = (cutoff - g["date"]).dt.days.values
            w = 0.5 ** (age / self.halflife) * self.delta ** back
            keep = w > 1e-3                              # matches too old to matter add only noise
            g, w = g[keep], w[keep]
        teams = sorted(set(self.teams[season]) | set(g["HomeTeam"]) | set(g["AwayTeam"]))
        pseudo = {}
        if prev:
            for t in self.teams[season]:
                if t not in self.teams[prev]:
                    pseudo[t] = (self.level[season], self.w0)
        if len(g) < 20:
            out = None
        else:
            r, hfa = solve(g, teams, w, pseudo, anchor=self.teams[season])
            out = (r, hfa, float(w.sum()))
        self._cache[key] = out
        return out


# ------------------------------------------------------------------------------------ evaluation
def build_rows(m, rater, eval_seasons):
    rows = []
    for k in HORIZONS:
        for r in m[m["season"].isin(eval_seasons)].itertuples():
            cutoff = r.date - pd.Timedelta(days=7 * k)
            snap = rater.ratings(cutoff, r.season)
            if snap is None:
                continue
            rt, hfa, wsum = snap
            rows.append({"season": r.season, "round": r.round, "date": r.date, "home": r.HomeTeam,
                         "away": r.AwayTeam, "horizon": k, "diff": rt[r.HomeTeam] - rt[r.AwayTeam],
                         "hfa": hfa, "wsum": wsum, "p_home": r.p_home, "p_draw": r.p_draw, "p_away": r.p_away})
    return pd.DataFrame(rows)


def feats(d):
    x = d["diff"].values
    return np.column_stack([x, np.abs(x), x ** 2])


def fit_map(train):
    """Multinomial logistic on soft targets: each match becomes three weighted rows (H/D/A)."""
    X = np.vstack([feats(train)] * 3)
    y = np.repeat([0, 1, 2], len(train))
    w = np.concatenate([train["p_home"].values, train["p_draw"].values, train["p_away"].values])
    return LogisticRegression(C=100.0, max_iter=2000).fit(X, y, sample_weight=w)


def predict_loso(rows):
    """Leave-one-season-out, per horizon: fit the diff -> probabilities map on the other seasons."""
    out = rows.copy()
    out["ph"] = out["pa"] = np.nan
    for k in HORIZONS:
        for s in out["season"].unique():
            te = (out["horizon"] == k) & (out["season"] == s)
            tr = (out["horizon"] == k) & (out["season"] != s)
            if tr.sum() < 100 or te.sum() == 0:
                continue
            clf = fit_map(out[tr])
            p = clf.predict_proba(feats(out[te]))
            out.loc[te, "ph"], out.loc[te, "pa"] = p[:, 0], p[:, 2]
    return out


def mae_win(d, ph="ph", pa="pa"):
    """Both perspectives: home P(win) and away P(win), like the pipeline's win / opponent-win."""
    e = np.concatenate([(d[ph] - d["p_home"]).abs().values, (d[pa] - d["p_away"]).abs().values])
    return float(np.nanmean(e))


def report(rows, label):
    d = rows.dropna(subset=["ph"])
    mean_h, mean_a = d["p_home"].mean(), d["p_away"].mean()
    print(f"\n[{label}]  {len(d)} match-horizons, seasons {sorted(d['season'].unique())}")
    print(f"{'horizon':<9}{'n':>6} | {'rating':>8}{'mean':>8}")
    for k in HORIZONS:
        s = d[d["horizon"] == k]
        if len(s):
            base = float(np.mean(np.concatenate([(mean_h - s['p_home']).abs(), (mean_a - s['p_away']).abs()])))
            print(f"F{k+1:<8}{len(s):>6} | {mae_win(s):>8.4f}{base:>8.4f}")
    print(f"{'all':<9}{len(d):>6} | {mae_win(d):>8.4f}")
    d = d.assign(phase=pd.cut(d["round"], [0, 3, 9, 20, 40], labels=["rounds 1-3", "4-9", "10-20", "21+"]))
    print("  by season phase (all horizons): " + "  ".join(
        f"{ph}: {mae_win(s):.4f} (n={len(s)})" for ph, s in d.groupby("phase", observed=True)))


def incumbent_on_archive(rows):
    """Incumbent outrights model on the 2025-26 archive matches, vs the rating at each horizon."""
    path = os.path.join(config.INPUTS_DIR, "historical_fixture_odds.csv")
    h = pd.read_csv(path)
    h = h[h["Season"] == "2025-2026"].dropna(subset=["home_title_odds", "away_title_odds"])
    for c in h.columns:
        if c.endswith("_odds"):
            h[c] = pd.to_numeric(h[c], errors="coerce")
    imp = lambda o: 1 / o / config.MARGIN_SEASON  # noqa: E731
    odds_cols = [f"{s}_{m}_odds" for s in ("home", "away") for m in ("title", "relegation", "top_6")]
    ok = h[odds_cols].notna().all(axis=1) & (h[odds_cols] > 1.0).all(axis=1)   # drop 0/1 sentinels
    # the archive holds duplicate (home, away) rows (re-archived weeks); keep the latest gameweek
    h = (h[ok].sort_values("Gameweek").drop_duplicates(subset=["home_team", "away_team"], keep="last")
         .copy())
    if h.empty:
        print("\n(no archive matches with full outrights - incumbent comparison skipped)")
        return
    ht, hr, h6 = imp(h["home_title_odds"]), imp(h["home_relegation_odds"]), imp(h["home_top_6_odds"])
    at, ar, a6 = imp(h["away_title_odds"]), imp(h["away_relegation_odds"]), imp(h["away_top_6_odds"])
    one = pd.Series(True, index=h.index)
    wh = model.win_pred(ht, hr, h6, at, ar, a6, one).clip(0, 1)
    wa = model.win_pred(at, ar, a6, ht, hr, h6, ~one).clip(0, 1)
    h["inc_h"], h["inc_a"] = wh.values, wa.values
    key = ["home", "away"]
    j = rows[(rows["season"] == "2025-2026")].dropna(subset=["ph"]).merge(
        h.rename(columns={"home_team": "home", "away_team": "away"})[key + ["inc_h", "inc_a"]], on=key)
    if j.empty:
        print("\n(archive matches did not join to football-data - check team names)")
        return
    n_matches = len(j[j["horizon"] == 1])
    print(f"\n[incumbent outrights vs rolling rating on {n_matches} archived 2025-26 matches with full outrights]")
    s1 = j[j["horizon"] == 1]
    print(f"  incumbent bias: home {(s1['inc_h'] - s1['p_home']).mean():+.3f}, away {(s1['inc_a'] - s1['p_away']).mean():+.3f}; "
          f"sample: {s1.iloc[0]['home']} v {s1.iloc[0]['away']} closing {s1.iloc[0]['p_home']:.2f}/{s1.iloc[0]['p_away']:.2f}, "
          f"incumbent {s1.iloc[0]['inc_h']:.2f}/{s1.iloc[0]['inc_a']:.2f}, rating {s1.iloc[0]['ph']:.2f}/{s1.iloc[0]['pa']:.2f}")
    print(f"{'horizon':<9}{'n':>5} | {'rating':>8}{'incumbent':>10}")
    for k in HORIZONS:
        s = j[j["horizon"] == k]
        if len(s):
            print(f"F{k+1:<8}{len(s):>5} | {mae_win(s):>8.4f}{mae_win(s, 'inc_h', 'inc_a'):>10.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--grid", action="store_true", help="search half-life / season discount / promoted weight")
    ap.add_argument("--halflife", type=float, default=60.0)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--w0", type=float, default=10.0)
    args = ap.parse_args()

    m = load_matches(args.seasons)
    seasons = sorted(m["season"].unique())
    eval_seasons = seasons[1:]                      # first season is history only
    print(f"{len(m)} matches, {len(seasons)} seasons; evaluating {eval_seasons}")

    if args.grid:
        print("\ngrid (pooled LOSO MAE over horizons 1-8):")
        best = None
        for hl, de, w0 in itertools.product([30, 60, 120, 240], [0.25, 0.5, 1.0], [5, 20]):
            rows = predict_loso(build_rows(m, Rater(m, hl, de, w0), eval_seasons))
            s = mae_win(rows.dropna(subset=["ph"]))
            print(f"  half-life {hl:>4}d  delta {de:.2f}  w0 {w0:>2}: {s:.4f}")
            if best is None or s < best[0]:
                best = (s, hl, de, w0)
        print(f"best: half-life {best[1]}d delta {best[2]} w0 {best[3]} -> {best[0]:.4f}")
        args.halflife, args.delta, args.w0 = best[1], best[2], best[3]

    rating = predict_loso(build_rows(m, Rater(m, args.halflife, args.delta, args.w0), eval_seasons))
    report(rating, f"rolling rating: half-life {args.halflife:g}d, season discount {args.delta}, promoted w0 {args.w0:g}")
    static = predict_loso(build_rows(m, Rater(m, 1e9, 1.0, args.w0, static=True), eval_seasons))
    report(static, "static prior-season rating (existing construction)")
    incumbent_on_archive(rating)

    out = os.path.join(config.OUTPUTS_DIR, "rolling_rating_eval.csv")
    rating.to_csv(out, index=False)
    print(f"\nrows -> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
