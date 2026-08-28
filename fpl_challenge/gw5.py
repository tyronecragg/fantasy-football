"""GW5 'The Shield' — DEFENSIVE CONTRIBUTIONS score 10 points instead of the usual 2.
Max 1 per club.

DefCon normally pays 2 points for hitting the threshold (defenders 10+ CBIT, midfielders
12+); this week it pays 10. So we add start x (10 - 2) x P(hit threshold) on top of normal
XP. The projections model DefCon for defenders and midfielders only (no forward DefCon),
so forwards get no boost. Captain then doubles the boosted total.

    python gw5.py
"""
import argparse

import challenge_core as cc

PER_HIT = 10.0
NORMAL = 2.0


def main(exclude=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-club", type=int, default=1)
    ap.add_argument("--exclude", nargs="*", default=[], metavar="NAME",
                    help="player names to remove from the pool (quote multi-word names)")
    args = ap.parse_args()

    df = cc.load_players()
    df = cc.apply_exclusions(df, list(exclude or []) + list(args.exclude))
    extra = cc.defcon_points(df, PER_HIT - NORMAL)   # the uplift over the 2 pts already in XP
    df["eff_xp"] = df[cc.XP_COL] + extra
    df["cap_bonus"] = df["eff_xp"]
    df["boosted"] = extra > 0.10

    title = ["FPL CHALLENGE  |  GW5 'The Shield'  |  defensive contributions score 10 (not 2)",
             "boost applies to defenders & midfielders (no forward DefCon in the model)"]
    cc.solve_and_report(df, args.max_per_club, title, "DEFCON x10")


if __name__ == "__main__":
    # Exclude players here (or pass --exclude on the command line). Multi-word
    # names are fine; matching is accent/case-insensitive.
    main(exclude=[])
