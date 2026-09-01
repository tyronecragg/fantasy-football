# -*- coding: utf-8 -*-
"""Team FIXTURE-DIFFICULTY outlook over a long horizon — for wildcard / chip-window planning.

    env/Scripts/python tools/fixture_outlook.py [--gw N] [--window 20]

Unlike the player XP pipeline (which stops at F8 because per-player projections that far out are
noise), TEAM strength is stable all season, so a team's win probability per fixture is a reliable,
useful signal well beyond F8. This reads `inputs/season_fixtures.csv` (the full 380-fixture list)
and the same season-strength model the pipeline uses (`team_model.season_probs` -> `model.win_pred`
off title/relegation/top-6 odds) to score every upcoming fixture for every team.

Output: teams ranked by their average win probability over the window (best RUNS at the top), plus a
ticker grid — each cell is the opponent (CAPS = home, lower = away) + an ease tier 1(hardest)-5(easiest).
Use it to spot who has a soft patch to target (a wildcard, or a first-half Triple Captain on a single),
and who to avoid. It does NOT model doubles/blanks (those aren't scheduled yet)."""
import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import ingest, team_model, model, config  # noqa: E402
from fpl_pipeline.names import apply_team_names  # noqa: E402

# Explicit 3-letter club codes — the last-word heuristic collapses "... City"/"... Town" teams.
FIRST_HALF_END = 19   # last GW of the first-half chip window (chips reset for GW20-38)

CODE = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE",
    "Brighton": "BHA", "Chelsea": "CHE", "Coventry City": "COV", "Crystal Palace": "CRY",
    "Everton": "EVE", "Fulham": "FUL", "Hull City": "HUL", "Ipswich Town": "IPS",
    "Leeds": "LEE", "Liverpool": "LIV", "Man City": "MCI", "Man Utd": "MUN",
    "Newcastle": "NEW", "Nott'm Forest": "NFO", "Spurs": "TOT", "Sunderland": "SUN",
}


def _code(team):
    if team not in CODE:                              # fallback for an unmapped / renamed club
        t = re.sub(r"[^A-Za-z ]", "", str(team)).replace(" ", "")
        CODE[team] = t[:3].upper()
    return CODE[team]


def _current_gw():
    fx = pd.read_csv(os.path.join(config.ROOT, "inputs", "fixtures.csv"), nrows=0).columns
    m = re.match(r"GW(\d+)", str(fx[1])) if len(fx) > 1 else None
    return int(m.group(1)) if m else 1


def build(start_gw, window):
    model.load_coefficients()
    sp = team_model.season_probs(ingest.load_inputs()).set_index("team")
    fx = pd.read_csv(os.path.join(config.ROOT, "inputs", "season_fixtures.csv"))
    fx["home_team"] = apply_team_names(fx["home_team"])
    fx["away_team"] = apply_team_names(fx["away_team"])
    fx = fx[(fx["gameweek"] >= start_gw) & (fx["gameweek"] < start_gw + window)]

    def probs(col):                                # attach a team's season strengths
        return (fx[col].map(sp["title"]), fx[col].map(sp["relegation"]), fx[col].map(sp["top6"]))
    ht, hr, h6 = probs("home_team")
    at, ar, a6 = probs("away_team")
    yes = pd.Series(True, index=fx.index)
    fx = fx.assign(
        win_home=model.win_pred(ht, hr, h6, at, ar, a6, yes),
        win_away=model.win_pred(at, ar, a6, ht, hr, h6, ~yes))

    rows = []                                      # one row per team-fixture (win = this team's, opp_win = opponent's)
    for _, r in fx.iterrows():
        rows.append({"gw": int(r["gameweek"]), "team": r["home_team"], "opp": r["away_team"], "venue": "H", "win": r["win_home"], "opp_win": r["win_away"]})
        rows.append({"gw": int(r["gameweek"]), "team": r["away_team"], "opp": r["home_team"], "venue": "A", "win": r["win_away"], "opp_win": r["win_home"]})
    d = pd.DataFrame(rows).dropna(subset=["win", "opp_win", "team"])
    return d[d["team"].isin(sp.index)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, default=None, help="first gameweek of the window (default: current F1 GW)")
    ap.add_argument("--window", type=int, default=None,
                    help=f"gameweeks to look ahead (default: through GW{FIRST_HALF_END}, the first-half chip window)")
    a = ap.parse_args()
    start = a.gw if a.gw is not None else _current_gw()
    window = a.window if a.window is not None else max(1, FIRST_HALF_END - start + 1)
    d = build(start, window)
    gws = sorted(d["gw"].unique())

    order = d.groupby("team")["win"].mean().sort_values(ascending=False)
    print(f"\nFixture outlook — GW{gws[0]}..GW{gws[-1]} ({len(gws)} GWs). Cell = opponent (CAPS=home, lower=away) + "
          f"this team's win % (higher = easier). Teams ranked by average win probability (best runs first).\n")
    header = "  " + f"{'team':<16}{'avg':>5}   " + "".join(f"{'GW'+str(g):<6}" for g in gws)
    print(header)
    for team in order.index:
        sub = d[d["team"] == team].set_index("gw")
        cells = ""
        for g in gws:
            if g in sub.index:
                r = sub.loc[g]
                if isinstance(r, pd.DataFrame):     # a double (shouldn't happen pre-schedule), take first
                    r = r.iloc[0]
                code = _code(r["opp"])
                cells += f"{(code if r['venue']=='H' else code.lower())+str(round(r['win']*100)):<6}"
            else:
                cells += f"{'--':<6}"
        print(f"  {team:<16}{order[team]*100:>4.0f}%  {cells}")
    print("\n  cell number = the TEAM's own modelled win probability % for that fixture (higher = easier; strength x")
    print("  fixture x venue). avg% = mean over the window, drives the ranking. Not real odds beyond F1/F2.")


if __name__ == "__main__":
    main()
