"""Fit the SHAPE of Betway's player-market margin, so longshots stop being overstated.

    python tools/fit_margin_power.py

THE PROBLEM. Two facts are both true and cannot be reconciled by a flat divisor:
  * summing implied P(1+ goal) over a team's players gives ~3.5x that team's expected goals
    (tools/margin_goals.py, measured against the clean-sheet market which de-margins exactly)
  * the favourite's price is already close to fair — Gyokeres at 1.91 is 52%, and a striker
    for a side expected to score 2.15 should be around 45-55%
Divide everything by 3.54 and Gyokeres becomes 14.8%, which is absurd. So essentially the
whole load sits on the longshots, and config.MARGIN_PLAYER = 1.05 — the same haircut at
every price — leaves unlikely scorers systematically OVERSTATED relative to likely ones.
That distortion is relative, so it does NOT cancel in the factor calculation.

THE MODEL. The standard fix is the power method: raise each raw probability to a power and
renormalise to the known total.

    p_i = lambda_team * (1/o_i)^n / sum_j (1/o_j)^n

n = 1 is proportional scaling (today's behaviour, a uniform load). n > 1 compresses long
prices harder than short ones. Identification needs two constraints and we have two:
the sum must equal lambda (from the clean-sheet market), and the FAVOURITE must come out
near fair, since bookmakers price their headline selections tightly. n is then solved so the
shortest price lands at its raw probability divided by MARGIN_FAV.

This tool only MEASURES n. It deliberately does not touch the pipeline: changing the margin
model moves every player's factor at once, so it must clear tools/backtest_projections.py
against archived outcomes before it is adopted — the same gate applied to every model change
(see the 2026-07-25 refit verdict, where an apparently better fit degraded projections
100-225%).
"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config  # noqa: E402

MARGIN_FAV = 1.05      # assumed load on the shortest price in a market


def solve_n(raw, total, margin_fav=MARGIN_FAV):
    """n such that the top-priced selection lands at raw_max / margin_fav."""
    raw = np.asarray(sorted(raw, reverse=True), dtype=float)
    target = raw[0] / margin_fav
    lo, hi = 0.2, 6.0
    for _ in range(80):
        n = (lo + hi) / 2
        w = raw ** n
        p_top = total * w[0] / w.sum()
        # p_top RISES with n (weight concentrates on the short prices), so if the
        # favourite is still below target we need MORE compression, not less.
        if p_top < target:
            lo = n
        else:
            hi = n
    return (lo + hi) / 2


def main():
    cs = pd.read_csv(os.path.join(config.SPORTSBET_DIR, "sportsbet_clean_sheet_odds.csv"))
    gs = pd.read_csv(os.path.join(config.SPORTSBET_DIR, "sportsbet_goalscorer_odds.csv"))
    roster = pd.read_csv(os.path.join(config.OUTPUTS_DIR, "01_fpl_players.csv"))
    team_of = dict(zip(roster["name"], roster["team"]))
    gs["team"] = gs["player_name"].map(team_of)

    print(f"{'fixture':<32}{'team':<16}{'lam':>6}{'n':>7}{'fav raw':>9}{'fav n=1':>9}{'fav fit':>9}")
    ns = []
    for match, grp in gs.groupby("match_id"):
        rows = cs[cs["match_name"] == match]
        lam = {}
        for r in rows.itertuples():
            if pd.notna(r.clean_sheet_yes) and pd.notna(r.clean_sheet_no):
                p = (1/r.clean_sheet_yes) / ((1/r.clean_sheet_yes) + (1/r.clean_sheet_no))
                lam[r.team_name] = -math.log(max(p, 1e-6))
        if len(lam) != 2:
            continue
        for team, sub in grp.groupby("team"):
            opp = [t for t in lam if t != team]
            if team not in lam or not opp:
                continue
            L = lam[opp[0]]                       # opponent's clean sheet -> this team's goals
            raw = (1 / sub["odds_decimal"]).to_numpy()
            if len(raw) < 8 or L <= 0:
                continue
            n = solve_n(raw, L)
            ns.append(n)
            w = np.sort(raw)[::-1] ** n
            fav_fit = L * w[0] / w.sum()
            print(f"{match[:30]:<32}{team:<16}{L:>6.2f}{n:>7.2f}"
                  f"{max(raw):>8.1%}{max(raw)/1.05:>9.1%}{fav_fit:>9.1%}")
    print(f"\nfitted n: mean {np.mean(ns):.2f}  median {np.median(ns):.2f}  "
          f"range {min(ns):.2f}-{max(ns):.2f}  (n=1 is today's flat behaviour)")

    # what it does across the price range, at the median n
    n = float(np.median(ns))
    print(f"\nEffect at n={n:.2f} on a representative team (lambda 2.15, 44 players priced):")
    sub = gs[(gs["match_id"] == "Arsenal vs. Coventry City")].sort_values("odds_decimal")
    raw = (1 / sub["odds_decimal"]).to_numpy()
    w = raw ** n
    p = 2.15 * w / w.sum()
    print(f"  {'player':<24}{'odds':>7}{'now /1.05':>11}{'power fit':>11}{'change':>9}")
    for i in list(range(4)) + list(range(len(sub) - 4, len(sub))):
        r = sub.iloc[i]
        now = (1 / r.odds_decimal) / 1.05
        print(f"  {r.player_name[:22]:<24}{r.odds_decimal:>7.2f}{now:>11.1%}{p[i]:>11.1%}"
              f"{p[i]/now - 1:>+8.0%}")
    print(f"\n  sum now /1.05 = {sum(raw)/1.05:.2f}   sum power fit = {p.sum():.2f}   "
          f"team lambda = 2.15")
    print("\nNOT APPLIED. Gate against tools/backtest_projections.py before adopting.")


if __name__ == "__main__":
    main()
