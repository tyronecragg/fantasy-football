"""GW4 'Derby Day' — players from Manchester City and Manchester United score DOUBLE.
Per-club limit expanded to THREE.

    python gw4.py
    python gw4.py --no-stack       # captained Man player counts x3 not x4
"""
import argparse

import challenge_core as cc

MAN_CLUBS = {"Man City", "Man Utd"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-stack", dest="stack", action="store_false")
    ap.add_argument("--max-per-club", type=int, default=3)
    args = ap.parse_args()

    df = cc.load_players()
    df["boosted"] = df["Team"].isin(MAN_CLUBS)
    df["eff_xp"] = df[cc.XP_COL] * df["boosted"].map({True: 2.0, False: 1.0})
    if args.stack:
        df["cap_bonus"] = df["eff_xp"]
    else:
        df["cap_bonus"] = df.apply(
            lambda r: r[cc.XP_COL] if r["boosted"] else r["eff_xp"], axis=1)

    seen = sorted(set(df.loc[df["boosted"], "Team"]))
    if set(seen) != MAN_CLUBS:
        print(f"  ! expected {MAN_CLUBS}, found {seen} — check team labels")
    title = ["FPL CHALLENGE  |  GW4 'Derby Day'  |  Man City & Man Utd score DOUBLE",
             f"boosted clubs in data: {', '.join(seen)}"]
    cc.solve_and_report(df, args.max_per_club, title, "MAN x2")


if __name__ == "__main__":
    main()
