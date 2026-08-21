"""Shared engine for the FPL Challenge weekly optimisers.

Every week uses the same base game:
  - Pick 1 goalkeeper + 5 outfielders in one of six allowed formations
    (GK implicit; outfield shape is DEF-MID-FWD): 1-1-3, 1-2-2, 1-3-1, 2-1-2, 2-2-1, 3-1-1.
  - No budget. A per-club cap (default 1, some weeks 3).
  - One captain scores DOUBLE.

What differs week to week is the scoring twist, which each gw*.py expresses as two
columns on the player frame before calling solve_and_report:
  eff_xp    effective expected points for the week (already includes any x2 doubling)
  cap_bonus the extra points if this player is captain (one more copy of their score)

With a per-club cap the pick is not separable by position, so each formation is solved
exactly as a small integer program.
"""
import os
import sys
import unicodedata

import numpy as np
import pandas as pd
import pulp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYERS_CSV = os.path.join(ROOT, "outputs", "13_players_master.csv")
INPUTS = os.path.join(ROOT, "inputs")
XP_COL = "F1 XP"

FORMATIONS = {
    "1-1-3": (1, 1, 3), "1-2-2": (1, 2, 2), "1-3-1": (1, 3, 1),
    "2-1-2": (2, 1, 2), "2-2-1": (2, 2, 1), "3-1-1": (3, 1, 1),
}

GOAL_POINTS = {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4}  # mirrors fpl_pipeline/model.py


def norm(s):
    """Accent- and case-insensitive key for name matching."""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()


def load_players():
    df = pd.read_csv(PLAYERS_CSV)
    df.columns = df.columns.str.strip()
    if XP_COL not in df.columns:
        sys.exit(f"'{XP_COL}' column not found in {PLAYERS_CSV}")
    df["Position"] = df["Position"].astype(str).str.upper().replace({"GKP": "GK"})
    df["_key"] = df["Player Name"].map(norm)
    return df.reset_index(drop=True)


def match_names(df, names):
    """Resolve a list of raw names to canonical Player Name values in df.
    Returns (matched set, unmatched list)."""
    by_key = df.set_index("_key")["Player Name"]
    matched, missed = set(), []
    for name in names:
        key = norm(name)
        if not key:
            continue
        if key in by_key.index:
            hit = by_key.loc[key]
            matched.add(hit if isinstance(hit, str) else hit.iloc[0])
        else:
            hit = df[df["_key"].str.contains(key.split()[-1], regex=False)]
            if len(hit):
                matched.add(hit.iloc[0]["Player Name"])
            else:
                missed.append(name)
    return matched, missed


def _series(df, col):
    return df[col].fillna(0.0) if col in df.columns else pd.Series(0.0, index=df.index)


def attacking_points(df):
    """Expected points from goals + assists this week (start-scaled), i.e. the amount
    that a 'goals and assists double' rule adds on top of normal XP."""
    pos = df["Position"]
    start = _series(df, "F1 Start")
    goals = pos.map(GOAL_POINTS).astype(float) * (
        _series(df, "F1 Score 1+") + _series(df, "F1 Score 2+") + _series(df, "F1 Score 3+"))
    assists = 3.0 * (_series(df, "F1 Assist") + _series(df, "F1 Assist 2+"))
    return start * (goals + assists)


def defcon_points(df, per_hit):
    """Expected defensive-contribution points at `per_hit` points per threshold hit
    (start-scaled). DEF uses the DEF probability, MID the MID probability; the model
    carries no forward DefCon, so forwards get 0."""
    pos = df["Position"]
    start = _series(df, "F1 Start")
    dc = np.where(pos == "DEF", _series(df, "F1 Defensive Contribution - DEF"),
                  np.where(pos == "MID", _series(df, "F1 Defensive Contribution - MID"), 0.0))
    return start * per_hit * pd.Series(dc, index=df.index)


def solve_formation(df, shape, max_per_club):
    """Exact ILP for one (DEF, MID, FWD) shape under the club cap.
    Returns (total, xi_dataframe, captain_row) or None if infeasible."""
    need = {"GK": 1, "DEF": shape[0], "MID": shape[1], "FWD": shape[2]}
    idx = list(df.index)
    prob = pulp.LpProblem("challenge", pulp.LpMaximize)
    pick = {i: pulp.LpVariable(f"pick_{i}", cat="Binary") for i in idx}
    capt = {i: pulp.LpVariable(f"capt_{i}", cat="Binary") for i in idx}

    prob += pulp.lpSum(df.loc[i, "eff_xp"] * pick[i] for i in idx) \
        + pulp.lpSum(df.loc[i, "cap_bonus"] * capt[i] for i in idx)

    for pos, k in need.items():
        pos_idx = [i for i in idx if df.loc[i, "Position"] == pos]
        prob += pulp.lpSum(pick[i] for i in pos_idx) == k
    prob += pulp.lpSum(capt[i] for i in idx) == 1
    for i in idx:
        prob += capt[i] <= pick[i]
    for team in df["Team"].unique():
        team_idx = [i for i in idx if df.loc[i, "Team"] == team]
        prob += pulp.lpSum(pick[i] for i in team_idx) <= max_per_club

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if prob.status != pulp.LpStatusOptimal:
        return None
    chosen = [i for i in idx if pick[i].varValue > 0.5]
    cap_i = next(i for i in idx if capt[i].varValue > 0.5)
    xi = df.loc[chosen].copy()
    total = xi["eff_xp"].sum() + df.loc[cap_i, "cap_bonus"]
    return total, xi, df.loc[cap_i]


def solve_and_report(df, max_per_club, title_lines, boost_tag):
    """Rank all six formations on the prepared frame and print the best XI.
    df must carry: eff_xp, cap_bonus, boosted (bool). boost_tag labels boosted rows."""
    print("=" * 72)
    for ln in title_lines:
        print(ln)
    print(f"max {max_per_club} per club   |   boosted players marked ({boost_tag})")
    print("=" * 72)

    results = []
    for name, shape in FORMATIONS.items():
        res = solve_formation(df, shape, max_per_club)
        if res:
            results.append((name, *res))
    if not results:
        sys.exit("No feasible formation (club cap too tight?).")
    results.sort(key=lambda r: r[1], reverse=True)

    print("\nFORMATION RANKING (projected total, captain applied)")
    for name, total, xi, cap in results:
        nb = int(xi["boosted"].sum())
        print(f"  {name:<7} {total:6.2f}   (captain {cap['Player Name']}, {nb} boosted in XI)")

    name, total, xi, cap = results[0]
    print("\n" + "=" * 72)
    print(f"BEST PICK  ->  {name}   projected {total:.2f} pts")
    print("=" * 72)
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    for _, row in xi.sort_values("Position", key=lambda c: c.map(order)).iterrows():
        tag = f" ({boost_tag})" if row["boosted"] else ""
        star = "  (C)" if row["Player Name"] == cap["Player Name"] else ""
        print(f"    {row['Position']:<4}{row['Player Name']:<24}{row['Team']:<15}"
              f"xp {row[XP_COL]:4.2f} -> {row['eff_xp']:5.2f}{tag}{star}")
    counts = cap["eff_xp"] + cap["cap_bonus"]
    print(f"\n  Captain: {cap['Player Name']}  "
          f"(base xp {cap[XP_COL]:.2f}, effective {cap['eff_xp']:.2f} -> counts {counts:.2f})")

    print(f"\nTOP BOOSTED PLAYERS BY EFFECTIVE XP (what drives the pick)")
    top = df[df["boosted"]].sort_values("eff_xp", ascending=False).head(12)
    for _, r in top.iterrows():
        print(f"    {r['Position']:<4}{r['Player Name']:<24}{r['Team']:<15}"
              f"xp {r[XP_COL]:4.2f} -> {r['eff_xp']:5.2f}")
    return results
