"""GW13 'Dribbling: 20' — +1 per successful dribble, plus a one-off +5 for 3 or more. One per club.

Dribbles in a match are counted as a Poisson around the player's rate for the minutes he plays:
    λ     = dribbles_per90 × MINS / 90
    extra = start × (λ + 5 × chance of 3+ dribbles)
dribbles_per90 is GW9's blend (this season + last season at half weight, FotMob via
build_dribbles.py). Refresh raw/fotmob_dribbles.txt and re-run build_dribbles.py first — the
current file dates from late October.
"""
import argparse
import math

import pandas as pd

import challenge_core as cc
import gw9

MINS = 85.0          # typical minutes for a starter
BONUS, THRESHOLD = 5.0, 3


def p_at_least(lam, k):
    return 1.0 - sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(k))


def compute(df):
    drib90, _, _ = gw9.build_rates(df)
    lam = drib90 * MINS / 90.0
    p3 = lam.map(lambda x: p_at_least(x, THRESHOLD))
    start = df["F1 Start"].fillna(0.0)
    df["drib90"], df["p3plus"] = drib90, p3
    extra = start * (lam + BONUS * p3)
    df["eff_xp"] = df[cc.XP_COL] + extra
    df["cap_bonus"] = df["eff_xp"]
    df["boosted"] = extra > 0.5
    return df


def main(confirmed_not_starting=None, confirmed_starting=None,
         current_team=None, captain=None, now=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-club", type=int, default=1)
    args = ap.parse_args()

    df = cc.load_players(gw=13)   # snapshots/gw13.csv if it exists, else the live master
    df = cc.confirm_not_starting(df, list(confirmed_not_starting or []))
    df = cc.confirm_starting(df, list(confirmed_starting or []))
    cc.set_live(df, current_team, captain, now)
    df = compute(df)

    title = ["FPL CHALLENGE  |  GW13 'Dribbling: 20'  |  +1 per dribble, +5 once for 3 or more",
             f"extra = start × (expected dribbles + 5 × chance of 3+), {MINS:g} mins per start"]
    cc.solve_and_report(df, args.max_per_club, title, "DRIB")


if __name__ == "__main__":
    # current_team / captain: once matches start, your six and captain for live locking.
    main(confirmed_not_starting=[], confirmed_starting=[], current_team=[], captain=None)
