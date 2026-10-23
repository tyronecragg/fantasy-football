"""GW9 'Trick or Treat' — +3 per successful dribble, −1 each time dispossessed.
One player per club.

    extra = start × (3 × dribbles_per90 − 1 × dispossessed_per90)

added to normal XP (captain doubles the boosted total). Both rates are THIS season:
  - dribbles_per90 from inputs/fpl_challenge_dribbles.csv (the in-game/stat leaderboard,
    matched to FPL names) — refresh it each week by re-pasting the list.
  - dispossessed_per90 from FPL-Core-Insights 2026-27 match stats, shrunk by SHRINK
    pseudo-minutes so small samples don't spike.
Players absent from a source default to 0 there.
"""
import argparse
import glob
import os

import pandas as pd

import challenge_core as cc

DRIBBLES_CSV = os.path.join(cc.INPUTS, "fpl_challenge_dribbles.csv")
DISP_DIR = os.path.join(cc.ROOT, "fpl_data", "FPL-Core-Insights", "data", "2026-2027",
                        "By Gameweek")
SHRINK = 270.0


def _dispossessed_per90():
    """norm(name) -> dispossessed per 90 this season (shrunk)."""
    frames = []
    for f in glob.glob(os.path.join(DISP_DIR, "GW*", "playermatchstats.csv")):
        try:
            s = pd.read_csv(f, usecols=lambda c: c in {"player_id", "minutes_played", "dispossessed"})
        except (pd.errors.EmptyDataError, ValueError):
            continue
        if len(s):
            frames.append(s)
    if not frames:
        return {}
    agg = pd.concat(frames, ignore_index=True).groupby("player_id").sum(numeric_only=True)
    pl_files = sorted(glob.glob(os.path.join(DISP_DIR, "GW*", "players.csv")))
    pl = pd.read_csv(pl_files[-1])
    pl["full"] = (pl["first_name"].fillna("") + " " + pl["second_name"].fillna("")).str.strip()
    id_name = dict(zip(pl["player_id"], pl["full"]))
    out = {}
    for pid, r in agg.iterrows():
        nm = id_name.get(pid)
        if nm and r["minutes_played"] > 0:
            out[cc.norm(nm)] = 90.0 * r["dispossessed"] / (r["minutes_played"] + SHRINK)
    return out


def build_rates():
    """norm(name) -> (dribbles_per90, dispossessed_per90), both this season."""
    if not os.path.exists(DRIBBLES_CSV):
        cc._big_warning([f"GW9: {os.path.basename(DRIBBLES_CSV)} not found — dribble points are 0.",
                         "Paste this season's dribble leaderboard and rebuild it."])
        drib = {}
    else:
        d = pd.read_csv(DRIBBLES_CSV)
        drib = {cc.norm(p): float(v) for p, v in zip(d["Player"], d["dribbles_per90"])}
    disp = _dispossessed_per90()
    return {k: (drib.get(k, 0.0), disp.get(k, 0.0)) for k in set(drib) | set(disp)}


def main(confirmed_not_starting=None, confirmed_starting=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-club", type=int, default=1)
    ap.add_argument("--confirmed-not-starting", nargs="*", default=[], metavar="NAME",
                    dest="cns", help="players confirmed benched/out — removed from the pool")
    ap.add_argument("--confirmed-starting", nargs="*", default=[], metavar="NAME",
                    dest="cs", help="players confirmed in the XI — start set to 1.0")
    args = ap.parse_args()

    df = cc.load_players()
    df = cc.confirm_not_starting(df, list(confirmed_not_starting or []) + list(args.cns))
    df = cc.confirm_starting(df, list(confirmed_starting or []) + list(args.cs))

    rates = build_rates()
    drib90 = df["_key"].map(lambda k: rates.get(k, (0.0, 0.0))[0])
    disp90 = df["_key"].map(lambda k: rates.get(k, (0.0, 0.0))[1])
    start = df["F1 Start"].fillna(0.0)
    extra = start * (3.0 * drib90 - 1.0 * disp90)

    df["eff_xp"] = df[cc.XP_COL] + extra
    df["cap_bonus"] = df["eff_xp"]
    df["boosted"] = extra > 0.10

    matched = int((drib90 + disp90 > 0).sum())
    title = ["FPL CHALLENGE  |  GW9 'Trick or Treat'  |  +3 per dribble, −1 per dispossession",
             f"this-season rates: dribbles from {os.path.basename(DRIBBLES_CSV)}, dispossessed "
             f"from match stats ({matched} players with data)"]
    cc.solve_and_report(df, args.max_per_club, title, "DRIB")


if __name__ == "__main__":
    # As lineups are confirmed, list players here (or pass --confirmed-* on the command
    # line). Multi-word names are fine; matching is accent/case-insensitive.
    main(confirmed_not_starting=[], confirmed_starting=[])
