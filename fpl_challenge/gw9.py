"""GW9 'Trick or Treat' — +3 per successful dribble, −1 each time the player is tackled.
One player per club.

    extra = start × (3 × dribbles_per90 − 1 × (dispossessed_per90 + failed_dribbles_per90))

"Tackled" has no column of its own: a player is tackled either while holding the ball
(dispossessed) or while trying to beat a man (a failed dribble). failed_dribbles_per90 =
dribbles_per90 × (100 / success_pct − 1), from the same FotMob leaderboard.

Rates blend THIS season with LAST season at half weight, weighted by minutes:
    rate = (this-season total + ½ × last-season total) ÷ (this-season mins + ½ × last-season mins)
so a regular starter's rate is mostly this season, while a player with few minutes so far (back
from injury, say) leans on last season. Sources:
  - dribbles + success %: inputs/fpl_challenge_dribbles.csv (this season) and
    inputs/fpl_challenge_dribbles_last.csv (last season), from FotMob via build_dribbles.py.
    A season only counts for a player if FotMob lists him that season.
  - dispossessed + minutes: FPL-Core-Insights match stats for both seasons (last season linked to
    this season's player ids by full name), shrunk by SHRINK pseudo-minutes.
Players absent from every source default to 0.
"""
import argparse
import glob
import os

import pandas as pd

import challenge_core as cc

DRIBBLES_CSV = os.path.join(cc.INPUTS, "fpl_challenge_dribbles.csv")
DRIBBLES_LAST_CSV = os.path.join(cc.INPUTS, "fpl_challenge_dribbles_last.csv")
CORE = os.path.join(cc.ROOT, "fpl_data", "FPL-Core-Insights", "data")
SEASON, LAST_SEASON = "2026-2027", "2025-2026"
LAST_WEIGHT = 0.5      # last season's minutes count half
SHRINK = 270.0


def _season_stats(season):
    """Full name -> (player_id, minutes, dispossessed) summed over a season's played gameweeks."""
    base = os.path.join(CORE, season, "By Gameweek")
    frames = []
    for f in glob.glob(os.path.join(base, "GW*", "playermatchstats.csv")):
        try:
            s = pd.read_csv(f, usecols=lambda c: c in {"player_id", "minutes_played", "dispossessed"})
        except (pd.errors.EmptyDataError, ValueError):
            continue
        if len(s):
            frames.append(s)
    pl_files = sorted(glob.glob(os.path.join(base, "GW*", "players.csv")))
    if not frames or not pl_files:
        return pd.DataFrame(columns=["player_id", "minutes_played", "dispossessed"])
    agg = pd.concat(frames, ignore_index=True).groupby("player_id").sum(numeric_only=True)
    pl = pd.read_csv(pl_files[-1])
    pl["full"] = (pl["first_name"].fillna("") + " " + pl["second_name"].fillna("")).str.strip()
    agg = agg.join(pl.set_index("player_id")["full"])
    return agg.reset_index()


def _fotmob(path):
    """name -> (dribbles_per90, failed_dribbles_per90) from a build_dribbles.py file."""
    if not os.path.exists(path):
        return {}
    d = pd.read_csv(path)
    pct = d["success_pct"] if "success_pct" in d.columns else pd.Series(100.0, index=d.index)
    return {p: (float(v), float(v) * (100.0 / float(q) - 1.0) if float(q) > 0 else 0.0)
            for p, v, q in zip(d["Player"], d["dribbles_per90"], pct)}


def build_rates(df):
    """Per-player (dribbles/90, failed dribbles/90, dispossessed/90), this season blended with last
    season at LAST_WEIGHT by minutes. Returns three Series aligned to df.index."""
    if not os.path.exists(DRIBBLES_CSV):
        cc._big_warning([f"GW9: {os.path.basename(DRIBBLES_CSV)} not found — dribble points are 0.",
                         "Run build_dribbles.py with this season's FotMob leaderboard."])
    fm1, fm0 = _fotmob(DRIBBLES_CSV), _fotmob(DRIBBLES_LAST_CSV)
    cur, last = _season_stats(SEASON), _season_stats(LAST_SEASON)
    # this season by player id (shared with the master); last season linked by full name via the
    # FULL squad list - not just players who have played, or someone yet to feature this season
    # (injured all autumn) would get last season's dribbles but lose last season's dispossessed
    pl_files = sorted(glob.glob(os.path.join(CORE, SEASON, "By Gameweek", "GW*", "players.csv")))
    if pl_files:
        roster = pd.read_csv(pl_files[-1])
        roster["full"] = (roster["first_name"].fillna("") + " " + roster["second_name"].fillna("")).str.strip()
        name_to_id = dict(zip(roster["full"], roster["player_id"]))
    else:
        name_to_id = dict(zip(cur["full"], cur["player_id"])) if len(cur) else {}
    m1 = cur.set_index("player_id")["minutes_played"].to_dict() if len(cur) else {}
    d1 = cur.set_index("player_id")["dispossessed"].to_dict() if len(cur) else {}
    m0, d0 = {}, {}
    for _, r in last.iterrows():
        pid = name_to_id.get(r.get("full"))
        if pid is not None:
            m0[pid], d0[pid] = r["minutes_played"], r["dispossessed"]

    drib, fail, disp = [], [], []
    for i in df.index:
        nm, pid = df.loc[i, "Player Name"], df.loc[i, "player_id"] if "player_id" in df.columns else None
        a, b = m1.get(pid, 0.0), LAST_WEIGHT * m0.get(pid, 0.0)
        # dribbles: blend only the seasons FotMob lists him in (absence isn't zero - it has a minutes floor)
        num_d = num_f = den = 0.0
        for rates, mins in ((fm1.get(nm), a), (fm0.get(nm), b)):
            if rates is not None:
                w = mins if mins > 0 else 1.0
                num_d += rates[0] * w
                num_f += rates[1] * w
                den += w
        drib.append(num_d / den if den else 0.0)
        fail.append(num_f / den if den else 0.0)
        mins = a + b
        disp.append(90.0 * (d1.get(pid, 0.0) + LAST_WEIGHT * d0.get(pid, 0.0)) / (mins + SHRINK)
                    if mins > 0 else 0.0)
    idx = df.index
    return pd.Series(drib, idx), pd.Series(fail, idx), pd.Series(disp, idx)


def main(confirmed_not_starting=None, confirmed_starting=None,
         current_team=None, captain=None, now=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-club", type=int, default=1)
    ap.add_argument("--confirmed-not-starting", nargs="*", default=[], metavar="NAME",
                    dest="cns", help="players confirmed benched/out — removed from the pool")
    ap.add_argument("--confirmed-starting", nargs="*", default=[], metavar="NAME",
                    dest="cs", help="players confirmed in the XI — start set to 1.0")
    args = ap.parse_args()

    df = cc.load_players(gw=9)   # snapshots/gw9.csv if it exists, else the live master
    df = cc.confirm_not_starting(df, list(confirmed_not_starting or []) + list(args.cns))
    df = cc.confirm_starting(df, list(confirmed_starting or []) + list(args.cs))
    cc.set_live(df, current_team, captain, now)   # mid-week: lock what's already been played

    drib90, fail90, disp90 = build_rates(df)
    tackled90 = disp90 + fail90          # tackled = dispossessed + failed dribbles
    start = df["F1 Start"].fillna(0.0)
    extra = start * (3.0 * drib90 - 1.0 * tackled90)

    df["eff_xp"] = df[cc.XP_COL] + extra
    df["cap_bonus"] = df["eff_xp"]
    df["boosted"] = extra > 0.10

    matched = int((drib90 + tackled90 > 0).sum())
    title = ["FPL CHALLENGE  |  GW9 'Trick or Treat'  |  +3 per dribble, −1 per time tackled",
             f"rates: this season + last season at {LAST_WEIGHT:g} weight, by minutes "
             f"({matched} players with data)"]
    cc.solve_and_report(df, args.max_per_club, title, "DRIB")


if __name__ == "__main__":
    # As lineups are confirmed, list players here (or pass --confirmed-* on the command
    # line). Multi-word names are fine; matching is accent/case-insensitive.
    # current_team / captain: once your team is set and matches start, list your six and captain —
    # anyone whose match has kicked off is locked in, and the armband can't move onto a played match.
    main(confirmed_not_starting=[], confirmed_starting=['Robin Roefs', 'Costinha', 'Jeremy Doku', 'Mohamed Belloumi'], current_team=[], captain=None)
