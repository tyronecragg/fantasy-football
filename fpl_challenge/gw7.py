"""GW7 'Away Days' — players in AWAY matches score DOUBLE. One player per club.

    python gw7.py
    python gw7.py --no-stack       # captained away player counts x3 not x4
"""
import argparse

import challenge_core as cc


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
    df["boosted"] = df["F1 Venue"].astype(str).str.upper().eq("A")
    df["eff_xp"] = df[cc.XP_COL] * df["boosted"].map({True: 2.0, False: 1.0})
    if args.stack:
        df["cap_bonus"] = df["eff_xp"]
    else:
        df["cap_bonus"] = df.apply(
            lambda r: r[cc.XP_COL] if r["boosted"] else r["eff_xp"], axis=1)

    n_away = int(df["boosted"].sum())
    title = ["FPL CHALLENGE  |  GW7 'Away Days'  |  players in away matches score DOUBLE",
             f"away players in pool: {n_away}"]
    cc.solve_and_report(df, args.max_per_club, title, "AWAY x2")


if __name__ == "__main__":
    # As lineups are confirmed, list players here (or pass --confirmed-* on the command
    # line). Multi-word names are fine; matching is accent/case-insensitive.
    main(confirmed_not_starting=[], confirmed_starting=[])
