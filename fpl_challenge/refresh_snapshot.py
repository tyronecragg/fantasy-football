"""Refresh a Challenge snapshot's odds for matches that haven't kicked off yet — without touching
the archive or any pipeline file.

For each still-to-play match it fetches fresh Betway prices (read-only, reusing tools/betway.py's
fetchers but never its write step) and compares them with the raw prices saved for that gameweek
(inputs/raw_odds/<season>/GWnn/). Each snapshot probability is scaled by how far its raw price
has moved (new 1/odds ÷ old 1/odds), so whatever the pipeline did to the raw price (margin,
longshot shrink) carries through unchanged. Then XP is recomputed with the same formula the
pipeline uses (bonus and anything not refreshed are carried over).

Refreshed: anytime goal (and 2+/3+ scaled with it), assist, clean sheet, win.
Not refreshed: cards, saves, concede ladders, DefCon (small, slow-moving).

    python refresh_snapshot.py 8            # every still-to-play match in snapshots/gw8.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import tools.betway as bw  # noqa: E402

GOAL_POINTS = {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4}


def _xp_pre(r):
    """Pre-bonus expected points — the pipeline's model.xp_pre, row-wise (verified to ~0.001)."""
    z = lambda c: 0.0 if pd.isna(r.get(c)) else float(r.get(c))
    pos = r["Position"]
    pts = 2.0 + GOAL_POINTS.get(pos, 0) * (z("F1 Score 1+") + z("F1 Score 2+") + z("F1 Score 3+"))
    pts += 3.0 * (z("F1 Assist") + z("F1 Assist 2+")) - z("F1 Yellow Card")
    if pos in ("GK", "DEF"):
        pts += 4.0 * z("F1 Clean Sheet") - 2.0 * (z("F1 Concede 2+ Goals") + z("F1 Concede 4+ Goals"))
    if pos == "MID":
        pts += 1.0 * z("F1 Clean Sheet")
    if pos == "GK":
        pts += z("F1 3+ Saves") + z("F1 6+ Saves")
    if pos == "DEF":
        pts += 2.0 * z("F1 Defensive Contribution - DEF")
    if pos == "MID":
        pts += 2.0 * z("F1 Defensive Contribution - MID")
    return z("F1 Start") * pts


def _prices(frame, key_col, odds_col):
    return {r[key_col]: float(r[odds_col]) for _, r in frame.iterrows()
            if pd.notna(r[odds_col]) and float(r[odds_col]) > 0}


def refresh(gameweek):
    snap_path = os.path.join(HERE, "snapshots", f"gw{gameweek}.csv")
    snap = pd.read_csv(snap_path, low_memory=False)
    season = str(snap["Season"].iloc[0]) if "Season" in snap.columns else "2026-2027"
    raw_dir = os.path.join(ROOT, "inputs", "raw_odds", season, f"GW{int(gameweek):02d}")
    old_g = pd.read_csv(os.path.join(raw_dir, "sportsbet_goalscorer_odds.csv"))
    old_a = pd.read_csv(os.path.join(raw_dir, "sportsbet_assist_odds.csv"))
    old_cs = pd.read_csv(os.path.join(raw_dir, "sportsbet_clean_sheet_odds.csv"))
    old_w = pd.read_csv(os.path.join(raw_dir, "sportsbet_win_draw_win_odds.csv"))

    # still-to-play matches in this snapshot = Betway events whose teams play each other here
    pairs = {frozenset((r["Team"], r["F1 Opponent"])) for _, r in snap.iterrows()}
    rows, wdw = bw.fixtures()
    live = [(e, lab, iso) for e, lab, iso in rows if frozenset(bw.split_teams(lab)) in pairs]
    if not live:
        print("no still-to-play snapshot matches listed on Betway — nothing to refresh")
        return
    print("refreshing:", ", ".join(lab for _, lab, _ in live))
    frames, _ = bw.collect(live, {lab: "f1" for _, lab, _ in live}, delay=1.5)

    new_g = _prices(frames.get("score1", pd.DataFrame(columns=["player_name", "odds_decimal"])),
                    "player_name", "odds_decimal")
    new_a = _prices(frames.get("assist", pd.DataFrame(columns=["player_name", "odds_decimal"])),
                    "player_name", "odds_decimal")
    og = _prices(old_g, "player_name", "odds_decimal")
    oa = _prices(old_a, "player_name", "odds_decimal")

    def cs_yes(frame):
        out = {}
        for _, r in frame.iterrows():
            y, n = r.get("clean_sheet_yes"), r.get("clean_sheet_no")
            if pd.notna(y) and pd.notna(n) and y > 0 and n > 0:
                out[r["team_name"]] = (1 / y) / (1 / y + 1 / n)
        return out
    new_cs, old_csp = cs_yes(frames.get("clean_sheet", pd.DataFrame())), cs_yes(old_cs)

    def win_p(frame):
        # only this gameweek's matches: the price files also hold next week's fixtures, and a
        # team plays twice across them (Liverpool v Brighton GW8, Liverpool v Arsenal GW9)
        out = {}
        for _, r in frame.iterrows():
            if frozenset((r.get("home_team"), r.get("away_team"))) not in pairs:
                continue
            h, d, a = r.get("home_win_odds"), r.get("draw_odds"), r.get("away_win_odds")
            if all(pd.notna(x) and x > 0 for x in (h, d, a)):
                s = 1 / h + 1 / d + 1 / a
                out[r["home_team"]], out[r["away_team"]] = (1 / h) / s, (1 / a) / s
        return out
    new_w, old_wp = win_p(wdw), win_p(old_w)

    teams = {t for _, lab, _ in live for t in bw.split_teams(lab)}
    m = snap["Team"].isin(teams)
    before = snap.loc[m, "F1 XP"].copy()
    changed = {"goal": 0, "assist": 0, "cs": 0, "win": 0}
    for i in snap.index[m]:
        r = snap.loc[i]
        nm, tm = r["Player Name"], r["Team"]
        if nm in new_g and nm in og:
            k = (1 / new_g[nm]) / (1 / og[nm])
            for c in ("F1 Score 1+", "F1 Score 2+", "F1 Score 3+"):
                if c in snap.columns and pd.notna(r[c]):
                    snap.at[i, c] = min(float(r[c]) * k, 0.95)
            changed["goal"] += 1
        if nm in new_a and nm in oa and pd.notna(r.get("F1 Assist")):
            snap.at[i, "F1 Assist"] = min(float(r["F1 Assist"]) * (1 / new_a[nm]) / (1 / oa[nm]), 0.95)
            changed["assist"] += 1
        if tm in new_cs and tm in old_csp and pd.notna(r.get("F1 Clean Sheet")):
            snap.at[i, "F1 Clean Sheet"] = min(float(r["F1 Clean Sheet"]) * new_cs[tm] / old_csp[tm], 0.95)
            changed["cs"] += 1
        if tm in new_w and tm in old_wp and pd.notna(r.get("F1 Win")):
            snap.at[i, "F1 Win"] = min(float(r["F1 Win"]) * new_w[tm] / old_wp[tm], 0.99)
            changed["win"] += 1
        # add only the CHANGE in points: the same formula on old vs new components, so any
        # difference between this formula and the pipeline's current model cancels out
        d = _xp_pre(snap.loc[i]) - _xp_pre(r)
        snap.at[i, "F1 XP"] = float(r["F1 XP"]) + d
        if pd.notna(r.get("F1 XP Pre")):
            snap.at[i, "F1 XP Pre"] = float(r["F1 XP Pre"]) + d

    snap.to_csv(snap_path, index=False)
    delta = (snap.loc[m, "F1 XP"] - before)
    print(f"updated {int(m.sum())} players in {len(teams)} teams "
          f"(goal {changed['goal']}, assist {changed['assist']}, clean sheet {changed['cs']}, win {changed['win']})")
    for tm in sorted(teams):
        if tm in new_w and tm in old_wp:
            print(f"  {tm:<15} win {old_wp[tm]*100:4.1f}% -> {new_w[tm]*100:4.1f}%"
                  + (f"   clean sheet {old_csp[tm]*100:4.1f}% -> {new_cs[tm]*100:4.1f}%"
                     if tm in new_cs and tm in old_csp else ""))
    big = delta.abs().sort_values(ascending=False).head(8)
    print("biggest XP moves:")
    for i in big.index:
        print(f"  {snap.loc[i, 'Player Name']:<24}{snap.loc[i, 'Team']:<14}"
              f"{before[i]:5.2f} -> {snap.loc[i, 'F1 XP']:5.2f}")
    print(f"wrote {snap_path}")


if __name__ == "__main__":
    refresh(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
