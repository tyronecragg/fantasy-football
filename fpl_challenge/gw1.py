"""GW1 'Instant Impact' — NEW SIGNINGS score double. Max 1 per club.

New-signings list: inputs/fpl_challenge_new_signings.csv ('Player' column), built from
the in-game filter toggle (the definitive source). Only this summer's arrivals qualify.

    python gw1.py                 # captained signing stacks to x4
    python gw1.py --no-stack      # captain does not re-double a signing (x3)
    python gw1.py --max-per-club 2
"""
import argparse
import os

import pandas as pd

import challenge_core as cc

SIGNINGS_CSV = os.path.join(cc.INPUTS, "fpl_challenge_new_signings.csv")


def main(exclude=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-stack", dest="stack", action="store_false")
    ap.add_argument("--max-per-club", type=int, default=1)
    ap.add_argument("--exclude", nargs="*", default=[], metavar="NAME",
                    help="player names to remove from the pool (quote multi-word names)")
    args = ap.parse_args()

    df = cc.load_players()
    df = cc.apply_exclusions(df, list(exclude or []) + list(args.exclude))
    flags = pd.read_csv(SIGNINGS_CSV)
    signings, missed = cc.match_names(df, flags["Player"].dropna().astype(str))
    df["boosted"] = df["Player Name"].isin(signings)
    df["eff_xp"] = df[cc.XP_COL] * df["boosted"].map({True: 2.0, False: 1.0})
    if args.stack:
        df["cap_bonus"] = df["eff_xp"]
    else:  # a signing's captain bonus is only its base xp (x3 total), not the doubled figure
        df["cap_bonus"] = df.apply(
            lambda r: r[cc.XP_COL] if r["boosted"] else r["eff_xp"], axis=1)

    title = ["FPL CHALLENGE  |  GW1 'Instant Impact'  |  new signings score DOUBLE",
             f"signings matched: {len(signings)}"
             + (f"   ! unmatched: {', '.join(missed)}" if missed else "")]
    cc.solve_and_report(df, args.max_per_club, title, "NEW x2")


if __name__ == "__main__":
    # Exclude players here (or pass --exclude on the command line). Multi-word
    # names are fine; matching is accent/case-insensitive.
    main(exclude=[])
