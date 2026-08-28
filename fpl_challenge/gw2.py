"""GW2 'Welcome Back' — players from the promoted trio (Coventry, Ipswich, Hull)
score DOUBLE. Per-club limit expanded to THREE.

    python gw2.py
    python gw2.py --no-stack       # captained promoted player counts x3 not x4
"""
import argparse

import challenge_core as cc

PROMOTED = {"Coventry City", "Ipswich Town", "Hull City"}


def main(exclude=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-stack", dest="stack", action="store_false")
    ap.add_argument("--max-per-club", type=int, default=3)
    ap.add_argument("--exclude", nargs="*", default=[], metavar="NAME",
                    help="player names to remove from the pool (quote multi-word names)")
    args = ap.parse_args()

    df = cc.load_players()
    df = cc.apply_exclusions(df, list(exclude or []) + list(args.exclude))
    df["boosted"] = df["Team"].isin(PROMOTED)
    df["eff_xp"] = df[cc.XP_COL] * df["boosted"].map({True: 2.0, False: 1.0})
    if args.stack:
        df["cap_bonus"] = df["eff_xp"]
    else:
        df["cap_bonus"] = df.apply(
            lambda r: r[cc.XP_COL] if r["boosted"] else r["eff_xp"], axis=1)

    seen = sorted(set(df.loc[df["boosted"], "Team"]))
    title = ["FPL CHALLENGE  |  GW2 'Welcome Back'  |  promoted trio score DOUBLE",
             f"promoted clubs in data: {', '.join(seen)}"]
    cc.solve_and_report(df, args.max_per_club, title, "PROMOTED x2")


if __name__ == "__main__":
    main(exclude=[])
