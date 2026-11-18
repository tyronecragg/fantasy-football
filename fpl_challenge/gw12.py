"""GW12 'Wonderkids' — players aged 23 or under at the FINAL GW12 deadline score DOUBLE. One per club.

    eff_xp = F1 XP × 2 if eligible, else F1 XP

Eligible = born after CUTOFF minus 24 years (i.e. not yet 24 on CUTOFF). CUTOFF is the last GW12
kickoff (Sun 29 Nov 2026, 18:30 SAST) — the strictest reading of 'final deadline', so nobody who
turns 24 mid-gameweek slips in. Birth dates: inputs/fpl_challenge_birthdates.csv (build_ages.py,
from FPL's API). Players with no birth date are treated as NOT eligible and listed.
Players whose 24th birthday falls in the gameweek window are printed so you can check in-game.
"""
import argparse
import os

import pandas as pd

import challenge_core as cc

BIRTH_CSV = os.path.join(cc.INPUTS, "fpl_challenge_birthdates.csv")
FIRST_DEADLINE = pd.Timestamp("2026-11-27")     # FPL GW12 deadline (date)
CUTOFF = pd.Timestamp("2026-11-29")             # last GW12 kickoff (date)


def compute(df):
    b = pd.read_csv(BIRTH_CSV)
    bd = pd.to_datetime(df["player_id"].map(dict(zip(b["player_id"], b["birth_date"]))), errors="coerce")
    turns24 = bd + pd.DateOffset(years=24)
    df["eligible"] = turns24 > CUTOFF
    edge = df[(turns24 >= FIRST_DEADLINE - pd.Timedelta(days=7)) & (turns24 <= CUTOFF + pd.Timedelta(days=7))]
    for _, r in edge.iterrows():
        print(f"  CHECK IN-GAME: {r['Player Name']} ({r['Team']}) turns 24 on "
              f"{(pd.Timestamp(bd[_]) + pd.DateOffset(years=24)).date()} -> "
              f"{'counted eligible' if r['eligible'] else 'counted NOT eligible'}")
    nodate = df[bd.isna() & (df[cc.XP_COL] > 2)]
    if len(nodate):
        print(f"  no birth date (treated as not eligible): {', '.join(nodate['Player Name'])}")
    df["eff_xp"] = df[cc.XP_COL] * df["eligible"].map({True: 2.0, False: 1.0})
    df["cap_bonus"] = df["eff_xp"]
    df["boosted"] = df["eligible"]
    return df


def main(confirmed_not_starting=None, confirmed_starting=None,
         current_team=None, captain=None, now=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-club", type=int, default=1)
    args = ap.parse_args()

    df = cc.load_players(gw=12)   # snapshots/gw12.csv if it exists, else the live master
    df = cc.confirm_not_starting(df, list(confirmed_not_starting or []))
    df = cc.confirm_starting(df, list(confirmed_starting or []))
    cc.set_live(df, current_team, captain, now)
    df = compute(df)

    title = ["FPL CHALLENGE  |  GW12 'Wonderkids'  |  aged 23 or under score DOUBLE",
             f"eligible = not yet 24 on {CUTOFF.date()} ({int(df['eligible'].sum())} players)"]
    cc.solve_and_report(df, args.max_per_club, title, "U24 x2")


if __name__ == "__main__":
    # current_team / captain: once matches start, your six and captain for live locking.
    main(confirmed_not_starting=[], confirmed_starting=[], current_team=[], captain=None)
