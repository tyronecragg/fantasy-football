"""GW8 'Legends' — players who have been at their club for at least five years score DOUBLE.
One player per club.

Legends list: inputs/fpl_challenge_legends.csv ('Player' column), built from the in-game
filter toggle (the definitive source) and matched to FPL names.

    python gw8.py
    python gw8.py --no-stack       # captained Legend counts x3 not x4
"""
import argparse
import os

import pandas as pd

import challenge_core as cc

LEGENDS_CSV = os.path.join(cc.INPUTS, "fpl_challenge_legends.csv")


def main(confirmed_not_starting=None, confirmed_starting=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-stack", dest="stack", action="store_false")
    ap.add_argument("--max-per-club", type=int, default=1)
    ap.add_argument("--confirmed-not-starting", nargs="*", default=[], metavar="NAME",
                    dest="cns", help="players confirmed benched/out — removed from the pool")
    ap.add_argument("--confirmed-starting", nargs="*", default=[], metavar="NAME",
                    dest="cs", help="players confirmed in the XI — start set to 1.0")
    args = ap.parse_args()

    df = cc.load_players()
    df = cc.confirm_not_starting(df, list(confirmed_not_starting or []) + list(args.cns))
    df = cc.confirm_starting(df, list(confirmed_starting or []) + list(args.cs))
    legends, missed = cc.match_names(df, pd.read_csv(LEGENDS_CSV)["Player"].dropna().astype(str))
    df["boosted"] = df["Player Name"].isin(legends)
    df["eff_xp"] = df[cc.XP_COL] * df["boosted"].map({True: 2.0, False: 1.0})
    if args.stack:
        df["cap_bonus"] = df["eff_xp"]
    else:
        df["cap_bonus"] = df.apply(
            lambda r: r[cc.XP_COL] if r["boosted"] else r["eff_xp"], axis=1)

    title = ["FPL CHALLENGE  |  GW8 'Legends'  |  5+ years at club score DOUBLE",
             f"legends matched: {len(legends)}"
             + (f"   ! unmatched: {', '.join(missed)}" if missed else "")]
    cc.solve_and_report(df, args.max_per_club, title, "LEGEND x2")


if __name__ == "__main__":
    # As lineups are confirmed, list players here (or pass --confirmed-* on the command
    # line). Multi-word names are fine; matching is accent/case-insensitive.
    main(confirmed_not_starting=[], confirmed_starting=[])
