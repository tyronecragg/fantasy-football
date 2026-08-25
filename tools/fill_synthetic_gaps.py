# -*- coding: utf-8 -*-
"""Backfill the current F1 player-prop markets from the previous gameweek's archived F2 projection,
for matches Betway hasn't priced — so attackers in unpriced fixtures aren't left on appearance
points only. Non-destructive and Betway-authoritative per match: real odds are kept, and synthetic
is added ONLY for teams absent from each market file (a team Betway priced is left exactly as-is,
so a teammate it omitted stays NA).

    env/Scripts/python tools/fill_synthetic_gaps.py [--season S] [--source-gw N]

`--source-gw N` (default: current-1 = the gameweek whose F2 projected the one now current) names the
archived snapshot to read: its `F2 <stat>` is this gameweek's F1 <stat>. Odds carry the standard
player margin so the pipeline's de-margining recovers the intended probability. Re-run
tools/betway.py afterwards as real markets open (it now upserts, so it won't wipe these)."""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fpl_pipeline import config, history, ingest  # noqa: E402
from tools.build_preseason_data import PLAYER_MARGIN  # noqa: E402

# market file -> the archived F2 column that projects this gameweek
PROP = {"sportsbet_goalscorer_odds.csv": "F2 Score 1+",
        "sportsbet_two_goals_odds.csv": "F2 Score 2+",
        "sportsbet_assist_odds.csv": "F2 Assist"}
BOOKING = ("sportsbet_booking_odds.csv", "F2 Yellow Card")
EPOCH = 1788030000


def _to_odds(p):
    return (1 / (p.clip(0.01, 0.95) * PLAYER_MARGIN)).round(2)


def _priced_teams(cur, keycol, tmap):
    """Teams Betway actually priced = the two MAJORITY teams of each priced match. Using the match
    (not a per-player roster lookup) makes it robust to loan players registered to one club but
    playing for another (e.g. Disasi @ Chelsea, on loan at Palace, would else flip Chelsea 'priced')."""
    team = cur["player_name"].map(tmap)
    priced = set()
    for _, idx in cur.groupby(keycol).groups.items():
        priced.update(team.loc[idx].value_counts().head(2).index)   # the 2 real teams of the match
    return priced


def _strip_synthetic(cur):
    """Drop rows this tool added before (idempotent re-runs): synthetic match_id >= 9e7, or date==EPOCH."""
    if "match_id" in cur.columns:
        mid = pd.to_numeric(cur["match_id"], errors="coerce")
        return cur[mid.isna() | (mid < 90000000)]
    return cur[pd.to_numeric(cur.get("date"), errors="coerce") != EPOCH]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=config.SEASON)
    ap.add_argument("--source-gw", type=int, default=None,
                    help="archived GW whose F2 = this GW's F1 (default: current-1)")
    a = ap.parse_args()
    src_gw = a.source_gw or ((history.infer_gameweek() or 2) - 1)

    tmap = ingest.load_fpl_players().drop_duplicates("name").set_index("name")["team"]
    arch = pd.read_csv(history.PLAYER_HISTORY_CSV, low_memory=False)
    src = arch[(arch["Season"] == a.season) & (pd.to_numeric(arch["Gameweek"], errors="coerce") == src_gw)]
    if src.empty:
        raise SystemExit(f"no archived rows for {a.season} GW{src_gw}")
    print(f"backfilling from {a.season} GW{src_gw} F2 projections (Betway-authoritative per match)\n")

    def gaps(col):
        s = src[["Player Name", "Team", "F2 Opponent", col]].copy()
        s[col] = pd.to_numeric(s[col], errors="coerce")
        return s.dropna(subset=[col])

    def emit(fname, col, keycol):
        path = os.path.join(config.SPORTSBET_DIR, fname)
        cur = _strip_synthetic(pd.read_csv(path))
        priced = _priced_teams(cur, keycol, tmap)
        add = gaps(col)
        add = add[~add["Team"].isin(priced) & ~add["Player Name"].isin(cur["player_name"])]
        if keycol == "match_id":
            rows = pd.DataFrame({"player_name": add["Player Name"].values,
                                 "match_id": [90000000 + i for i in range(len(add))],
                                 "odds_decimal": _to_odds(add[col]).values})
        else:
            rows = pd.DataFrame({"match_name": (add["Team"] + " v " + add["F2 Opponent"]).values,
                                 "date": EPOCH, "player_name": add["Player Name"].values,
                                 "odds_decimal": _to_odds(add[col]).values})
        pd.concat([cur, rows[cur.columns]], ignore_index=True).to_csv(path, index=False)
        print(f"  {fname:<38} +{len(rows):>3} synthetic (teams: {add['Team'].nunique()})")

    for fname, col in PROP.items():
        emit(fname, col, "match_id")
    emit(BOOKING[0], BOOKING[1], "match_name")


if __name__ == "__main__":
    main()
