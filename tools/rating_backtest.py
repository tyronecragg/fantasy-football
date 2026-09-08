# -*- coding: utf-8 -*-
"""Does a better fixture-strength prediction improve the F2-F8 PLAYER projections?

    env/Scripts/python tools/rating_backtest.py [--halflife 60 --delta 0.5 --w0 10]

Re-runs tools/backtest_projections on the 2025-26 archive twice: once with the incumbent
win_pred / opp_win_pred (season outrights) and once with the same factor x baseline machinery fed
win probabilities from the ROLLING odds-implied rating (tools/rolling_rating.py). The rating is
solved from closing prices BEFORE gameweek M's first kick-off (what we would have known at the
deadline), and mapped to probabilities with the per-horizon multinomial fitted on the OTHER
seasons, so 2025-26 is out-of-sample for the map.

Scores every stat per horizon: baseline MAE (incumbent) vs rating-fed MAE, plus the deployed
F1-blend of each, so the number that matters (what the optimiser consumes) is compared directly.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fpl_pipeline import config  # noqa: E402
from tools.backtest_projections import STATS, build_pairs, load_archive  # noqa: E402
from tools.build_team_ratings import DEFAULT_SEASONS  # noqa: E402
from tools.rolling_rating import HORIZONS, Rater, build_rows, feats, fit_map, load_matches  # noqa: E402

SEASON = "2025-2026"


def gw_dates(archive, matches, season=SEASON):
    """Gameweek -> first kick-off date, by joining the archive's F1 fixtures to football-data."""
    a = archive[archive["Season"] == season]
    a = a[a["F1 Venue"] == "H"][["Gameweek", "Team", "F1 Opponent"]].drop_duplicates()
    m = matches[matches["season"] == season][["HomeTeam", "AwayTeam", "date"]]
    j = a.merge(m, left_on=["Team", "F1 Opponent"], right_on=["HomeTeam", "AwayTeam"])
    return j.groupby("Gameweek")["date"].min().to_dict()


def make_win_fn(rater, maps, dates_by_season, scale=1.0):
    """(pair, m, n, home_n, season) -> (win, opp) from the rating at GW m's deadline, for the GW n
    fixture. dates_by_season: {season: {gw: first kick-off}}.
    scale: multiply the three-way win probabilities to land on the archive's 1/odds/1.03 scale
    (tested 2026-09-02: 1.025 made every stat slightly WORSE - leave at 1)."""
    def date_of(season, gw):
        """First kick-off of gw; when the archive week didn't join, step 7 days from the nearest known."""
        dates = dates_by_season.get(season, {})
        if not dates:
            return None
        if gw in dates and pd.notna(dates[gw]):
            return dates[gw]
        ref = min(dates, key=lambda g: abs(g - gw))
        return dates[ref] + pd.Timedelta(days=7 * (gw - ref))

    def fn(pair, m, n, home_n, season=SEASON):
        cutoff = date_of(season, m)
        if cutoff is None:
            return None
        k = int(round((date_of(season, n) - cutoff).days / 7))
        k = min(max(k, 1), max(HORIZONS))
        snap = rater.ratings(cutoff, season)
        if snap is None or k not in maps:
            return None
        rt, _, _ = snap
        own = pair["Team"].map(rt)
        opp = pair["F1 Opponent"].map(rt)
        diff = np.where(home_n, own - opp, opp - own)         # always home - away
        d = pd.DataFrame({"diff": np.nan_to_num(diff, nan=0.0)})   # unknown teams masked below
        p = maps[k].predict_proba(feats(d))                   # [P(home), P(draw), P(away)]
        win = np.where(home_n, p[:, 0], p[:, 2]) * scale
        lose = np.where(home_n, p[:, 2], p[:, 0]) * scale
        return (pd.Series(win, index=pair.index).where(own.notna() & opp.notna()),
                pd.Series(lose, index=pair.index).where(own.notna() & opp.notna()))
    return fn


# Grid-best rating settings (tools/rolling_rating.py --grid + shorter half-lives, 2026-09-02):
# half-life 30 days, no extra season discount (old matches decay anyway), promoted prior weight 2.
BEST = dict(halflife=30.0, delta=1.0, w0=2.0)
MAP_SEASONS_EXCLUDED = ("2025-2026", "2026-2027")   # archive seasons: the map must be out-of-sample for them


def build_rater(halflife=BEST["halflife"], delta=BEST["delta"], w0=BEST["w0"], seasons=None):
    """Rater over every cached football-data season (incl. the current one) + per-horizon maps
    fitted on seasons that are NOT archive seasons, so archive-based backtests stay out-of-sample."""
    seasons = seasons or (DEFAULT_SEASONS + ["2627"])
    matches = load_matches(seasons)
    rater = Rater(matches, halflife, delta, w0)
    fit_on = [s for s in sorted(matches["season"].unique()) if s not in MAP_SEASONS_EXCLUDED][1:]
    rows = build_rows(matches, rater, fit_on)
    maps = {k: fit_map(rows[rows["horizon"] == k]) for k in HORIZONS}
    print(f"rating maps fitted on {fit_on} ({len(rows)} match-horizons)")
    return rater, maps, matches


def rating_win_fn(archive):
    """A win_fn for build_pairs covering every archive season, built on fpl_pipeline.rating's
    RatingSource - the SAME class, maps and parameters the pipeline serves with, so training-time
    `rating_win` / `rating_opp` features equal what projection_serving computes live (zero skew).
    The GW deadline dates come from football-data (first kick-off of each archived gameweek)."""
    from fpl_pipeline import rating as rating_mod
    src = rating_mod.RatingSource(refresh=False)
    seasons = [s for s in archive["Season"].unique() if s in set(src.m["season"])]
    dates = {s: gw_dates(archive, src.m, s) for s in seasons}
    dates = {s: d for s, d in dates.items() if d}

    def date_of(season, gw):
        d = dates.get(season, {})
        if not d:
            return None
        if gw in d and pd.notna(d[gw]):
            return d[gw]
        ref = min(d, key=lambda g: abs(g - gw))
        return d[ref] + pd.Timedelta(days=7 * (gw - ref))

    def fn(pair, m, n, home_n, season=SEASON):
        cutoff = date_of(season, m)
        if cutoff is None:
            return None
        k = min(max(int(round((date_of(season, n) - cutoff).days / 7)), 1), max(HORIZONS))
        fr = src.at(cutoff, season)
        if fr is None:
            return None
        win, opp = fr.win_probs(pair["Team"], pair["F1 Opponent"], home_n, k)
        xg_for, xg_against = fr.expected_goals(pair["Team"], pair["F1 Opponent"], home_n, k)
        return win, opp, {"rating_xg_for": xg_for, "rating_xg_against": xg_against}
    return fn


def mae(a, b):
    a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    ok = a.notna() & b.notna()
    return float((a[ok] - b[ok]).abs().mean()) if ok.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--halflife", type=float, default=BEST["halflife"])
    ap.add_argument("--delta", type=float, default=BEST["delta"])
    ap.add_argument("--w0", type=float, default=BEST["w0"])
    ap.add_argument("--scale", type=float, default=1.0, help="win-prob scale factor (1.025 ~ the 1/odds/1.03 scale)")
    args = ap.parse_args()

    rater, maps, matches = build_rater(args.halflife, args.delta, args.w0)
    archive = load_archive()
    dates = {SEASON: gw_dates(archive, matches, SEASON)}
    print(f"gameweek deadlines from football-data: GW{min(dates[SEASON])}..GW{max(dates[SEASON])}")
    base = build_pairs(archive[archive["Season"] == SEASON])
    rated = build_pairs(archive[archive["Season"] == SEASON],
                        win_fn=make_win_fn(rater, maps, dates, scale=args.scale))
    key = ["stat", "horizon", "gw_from", "gw_to", "player"]
    j = base.merge(rated[key + ["predicted", "win_hat", "opp_hat"]], on=key, suffixes=("", "_r"))
    j = j[j["valid_opp"] & j["predicted"].notna() & j["predicted_r"].notna()]

    print(f"\n{len(j)} pairs. MAE vs the odds that materialised; 'blend' = deployed F1-blend where one exists.")
    print(f"{'stat':<12}{'hz':>4}{'n':>6} | {'win MAE inc':>12}{'win MAE rat':>12} | {'base':>8}{'rating':>8}{'Δ%':>7} | {'blend':>8}{'bl+rat':>8}{'Δ%':>7}")
    print("-" * 104)
    for stat in STATS:
        w = config.PROJECTION_BLEND.get(stat, 1.0)
        s_all = j[j["stat"] == stat]
        for k in sorted(s_all["horizon"].unique()) + ["all"]:
            s = s_all if k == "all" else s_all[s_all["horizon"] == k]
            pers = pd.to_numeric(s["persistence"], errors="coerce")
            b, r = mae(s["predicted"], s["actual"]), mae(s["predicted_r"], s["actual"])
            bb = mae(w * s["predicted"] + (1 - w) * pers, s["actual"])
            br = mae(w * s["predicted_r"] + (1 - w) * pers, s["actual"])
            wi = mae(s["win_hat"], s["actual_win"])
            wr = mae(s["win_hat_r"], s["actual_win"])
            hz = "all" if k == "all" else f"F{int(k)+1}"
            print(f"{stat:<12}{hz:>4}{len(s):>6} | {wi:>12.4f}{wr:>12.4f} | {b:>8.4f}{r:>8.4f}{(b-r)/b*100:>+6.1f}% | "
                  f"{bb:>8.4f}{br:>8.4f}{(bb-br)/bb*100:>+6.1f}%")
        print()


if __name__ == "__main__":
    main()
