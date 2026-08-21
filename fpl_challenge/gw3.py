"""GW3 'All Out Attack' — GOALS and ASSISTS score DOUBLE (for everyone). Max 1 per club.

This changes the points themselves, not a set of players: we add one more copy of each
player's expected goal + assist points on top of normal XP. Appearance, clean sheets,
DefCon, saves, cards and bonus are unchanged. Captain then doubles the whole boosted
total, so a captained goalscorer's goals count x4.

    python gw3.py
"""
import argparse

import challenge_core as cc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-club", type=int, default=1)
    args = ap.parse_args()

    df = cc.load_players()
    extra = cc.attacking_points(df)                 # points from goals+assists this week
    df["eff_xp"] = df[cc.XP_COL] + extra            # doubling = add one more copy
    df["cap_bonus"] = df["eff_xp"]                  # captain doubles the boosted total
    df["boosted"] = extra > 0.10

    title = ["FPL CHALLENGE  |  GW3 'All Out Attack'  |  goals & assists score DOUBLE",
             "boost = extra copy of each player's expected goal + assist points"]
    cc.solve_and_report(df, args.max_per_club, title, "G/A x2")


if __name__ == "__main__":
    main()
