"""Build GW11 'Make the Call' inputs: how often each player comes on as a substitute, and what he
scores when he does — from FPL's own per-gameweek data (FPL-Core-Insights player_gameweek_stats).

A sub appearance is a gameweek with minutes > 0 and starts == 0; its FPL points are that week's
event_points. Per player and season:
  gws         gameweeks he was registered (rows present)
  starts      starts
  sub_apps    appearances off the bench
  sub_mins    minutes in those appearances
  sub_pts     FPL points in those appearances
Last season is linked to this season's player ids by full name (ids change each season).
Also writes position-level averages (used as priors for small samples) to the same file as rows
with player_id = -1.

    python build_subs.py
"""
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORE = os.path.join(ROOT, "fpl_data", "FPL-Core-Insights", "data")
OUT = os.path.join(ROOT, "inputs", "fpl_challenge_subs.csv")
SEASONS = ("2026-2027", "2025-2026")
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def season_rows(season):
    frames = []
    for f in glob.glob(os.path.join(CORE, season, "By Gameweek", "GW*", "player_gameweek_stats.csv")):
        g = pd.read_csv(f)
        if g["minutes"].sum() <= 0:          # unplayed gameweek (pre-filled zeros)
            continue
        frames.append(g)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d["full"] = (d["first_name"].fillna("") + " " + d["second_name"].fillna("")).str.strip()
    d["sub"] = (d["minutes"] > 0) & (d["starts"] == 0)
    agg = d.groupby(["id", "full"]).apply(lambda x: pd.Series({
        "gws": len(x),
        "starts": int(x["starts"].sum()),
        "sub_apps": int(x["sub"].sum()),
        "sub_mins": int(x.loc[x["sub"], "minutes"].sum()),
        "sub_pts": int(x.loc[x["sub"], "event_points"].sum()),
    }), include_groups=False).reset_index()
    return agg


def main():
    cur, last = season_rows(SEASONS[0]), season_rows(SEASONS[1])
    master = pd.read_csv(os.path.join(ROOT, "outputs", "players_master.csv"))
    master.columns = master.columns.str.strip()
    pos_of = dict(zip(master["player_id"], master["Position"].astype(str).str.upper().replace({"GKP": "GK"})))

    # this season keyed by its id; last season linked by full name to this season's id
    name_to_id = dict(zip(cur["full"], cur["id"]))
    rows = []
    for season, d in ((SEASONS[0], cur), (SEASONS[1], last)):
        for _, r in d.iterrows():
            pid = r["id"] if season == SEASONS[0] else name_to_id.get(r["full"])
            if pid is None:
                continue
            rows.append({"player_id": int(pid), "name": r["full"], "season": season,
                         "position": pos_of.get(int(pid), ""),
                         **{k: int(r[k]) for k in ("gws", "starts", "sub_apps", "sub_mins", "sub_pts")}})
    out = pd.DataFrame(rows)

    # position priors from both seasons: sub-on rate when not starting, points per sub appearance
    pri = []
    for pos, g in out[out["position"] != ""].groupby("position"):
        nonstart = (g["gws"] - g["starts"]).sum()
        pri.append({"player_id": -1, "name": f"PRIOR {pos}", "season": "both", "position": pos,
                    "gws": int(g["gws"].sum()), "starts": int(g["starts"].sum()),
                    "sub_apps": int(g["sub_apps"].sum()), "sub_mins": int(g["sub_mins"].sum()),
                    "sub_pts": int(g["sub_pts"].sum())})
        print(f"  prior {pos}: comes on in {g['sub_apps'].sum() / max(nonstart, 1):.0%} of non-starts, "
              f"{g['sub_pts'].sum() / max(g['sub_apps'].sum(), 1):.2f} pts per sub appearance")
    out = pd.concat([out, pd.DataFrame(pri)], ignore_index=True)
    out.to_csv(OUT, index=False)
    n_cur = (out["season"] == SEASONS[0]).sum()
    print(f"wrote {OUT}: {len(out)} rows ({n_cur} this season)")


if __name__ == "__main__":
    sys.exit(main())
