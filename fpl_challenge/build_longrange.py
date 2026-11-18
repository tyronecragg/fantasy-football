"""Build GW10 'Spectacular' inputs: per-player shots from OUTSIDE the box, from Understat.

Understat serves every shot with its pitch position (X = fraction of the pitch length toward the
goal being attacked, Y = fraction of the width), result and season. A shot is outside the box when
it is further out than the box edge or wider than the box. Penalties are always inside.

    python build_longrange.py            # fetch (cached) and write the CSV
    python build_longrange.py --refresh  # ignore the cache and re-fetch every player

Writes inputs/fpl_challenge_longrange.csv, one row per player and season (this + last), with
minutes, outside-box shots, shots on target, goals and xG. Shot histories are cached in
raw/understat_shots.json so re-runs only fetch players not already cached.
"""
import argparse
import html
import json
import os
import sys
import time

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(HERE, "raw", "understat_shots.json")
OUT = os.path.join(ROOT, "inputs", "fpl_challenge_longrange.csv")
SEASONS = ("2026", "2025")          # Understat labels a season by its starting year
H = {"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"}

# Box: 16.5 m deep on a ~105 m pitch, 40.3 m wide on a ~68 m pitch (Understat uses 0-1 coordinates)
BOX_X = 1 - 16.5 / 105              # ~0.843: shots with X below this are outside the box
BOX_Y = (0.5 - 20.15 / 68, 0.5 + 20.15 / 68)   # ~(0.204, 0.796)
ON_TARGET = {"Goal", "SavedShot"}


def league_data(season):
    """(players, Premier League team titles) for one season."""
    url = f"https://understat.com/getLeagueData/EPL/{season}"
    r = requests.get(url, headers={**H, "Referer": f"https://understat.com/league/EPL/{season}"}, timeout=30)
    r.raise_for_status()
    d = r.json()
    return d["players"], {t["title"] for t in d["teams"].values()}


def player_shots(pid):
    url = f"https://understat.com/getPlayerData/{pid}"
    r = requests.get(url, headers={**H, "Referer": f"https://understat.com/player/{pid}"}, timeout=30)
    r.raise_for_status()
    return r.json().get("shots", [])


def outside_box(s):
    if s.get("situation") == "Penalty":
        return False
    x, y = float(s["X"]), float(s["Y"])
    return x < BOX_X or y < BOX_Y[0] or y > BOX_Y[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    cache = {}
    if os.path.exists(CACHE) and not args.refresh:
        cache = json.load(open(CACHE, encoding="utf-8"))

    # minutes per season, and the current-season roster to fetch shots for
    minutes, names, teams, epl_teams = {}, {}, {}, {}
    for season in SEASONS:
        players, epl_teams[season] = league_data(season)
        for p in players:
            minutes[(p["id"], season)] = float(p["time"])
            names.setdefault(p["id"], html.unescape(p["player_name"]))
            if season == SEASONS[0]:
                teams[p["id"]] = p["team_title"]
    current = [pid for pid in teams]
    todo = [pid for pid in current if pid not in cache]
    print(f"{len(current)} current players; fetching {len(todo)} (cached {len(current) - len(todo)})")
    for k, pid in enumerate(todo, 1):
        try:
            cache[pid] = player_shots(pid)
        except Exception as e:                       # keep going; report at the end
            print(f"  ! {names.get(pid, pid)}: {e}")
            continue
        if k % 50 == 0:
            print(f"  {k}/{len(todo)}")
            json.dump(cache, open(CACHE, "w", encoding="utf-8"))
        time.sleep(0.35)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(cache, open(CACHE, "w", encoding="utf-8"))

    rows = []
    for pid in current:
        shots = cache.get(pid, [])
        for season in SEASONS:
            mins = minutes.get((pid, season), 0.0)
            # Premier League shots only: a player's history includes other leagues (Ligue 1 etc.),
            # but minutes are Premier League minutes, so foreign shots would inflate the rate
            ob = [s for s in shots if s.get("season") == season and outside_box(s)
                  and s.get("h_team") in epl_teams[season] and s.get("a_team") in epl_teams[season]]
            if mins <= 0 and not ob:
                continue
            rows.append({"understat_id": pid, "player": names[pid], "team": teams[pid], "season": season,
                         "minutes": mins, "obox_shots": len(ob),
                         "obox_sot": sum(s["result"] in ON_TARGET for s in ob),
                         "obox_goals": sum(s["result"] == "Goal" for s in ob),
                         "obox_xg": round(sum(float(s["xG"]) for s in ob), 3)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    cur = out[out["season"] == SEASONS[0]]
    print(f"wrote {OUT}: {len(out)} rows ({len(cur)} this season)")
    allshots = [s for v in cache.values() for s in v if s.get("season") in SEASONS
                and s.get("h_team") in epl_teams[s["season"]] and s.get("a_team") in epl_teams[s["season"]]]
    share = sum(outside_box(s) for s in allshots) / max(len(allshots), 1)
    goals = [s for s in allshots if s["result"] == "Goal"]
    gshare = sum(outside_box(s) for s in goals) / max(len(goals), 1)
    print(f"sanity: {share:.0%} of shots and {gshare:.0%} of goals are from outside the box "
          f"(Premier League norm is roughly 35-45% of shots and 10-15% of goals)")


if __name__ == "__main__":
    sys.exit(main())
