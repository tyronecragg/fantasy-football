"""Extend the pipeline's fixture projections across the whole first half (GW2-19) for planning.

The production pipeline projects F1-F8 (GW2-GW9): F1/F2 on REAL odds, F3-F8 model-predicted. This
tool keeps those exactly and extends the SAME model machinery (win_pred / baseline / xp_pre) out to
GW19, so we have one best-current-estimate view of:
  1. upcoming FIXTURE MISMATCHES  - per-team win probability, GW2-19 (a fixture-ease matrix), and
  2. where the CURRENT SQUAD STRUGGLES - owned players' projected XP-Pre per GW, with the trough weeks.

It is standalone and read-only: it does NOT touch the pipeline/optimiser (which stay on the validated
8-fixture horizon). What GW9-19 share with F3-F8 vs where they degrade:
  - WIN PROBABILITY is the SAME model as F3-F8 (win_pred from team strength + venue; F3-F8 are predicted
    too, not market-based). So the fixture-ease matrix past F8 is as valid as F4-F9 - only F1/F2 use real
    odds anywhere. This is why the transfer-target view (fixture-ease driven) is on solid ground.
  - Degradations past F8: the defensive GBM projection MODELS (clean_sheet/concede2/saves3) and the
    F1-odds blend are not applied (validated only to F8 - pure factor x baseline there), and start prob
    is held at its GW9 (F8) steady-state. These touch player XP more than the win-prob fixture read.
  - XP is XP-Pre (pre-bonus), used consistently across GW2-19 so weeks compare like-for-like.

    python tools/fixture_horizon.py [--from-gw 8 --to-gw 15]   # matrices + transfer targets over the window
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config, model, players as P  # noqa: E402

MASTER = os.path.join(config.OUTPUTS_DIR, "13_players_master.csv")
FIRST_HALF_END = 19
STAT_FACTORS = {  # master factor column -> baseline stat key
    "score1": "Score 1+ Factor", "assist": "Assist Factor", "yellow": "F1 Yellow Card Factor",
    "clean_sheet": "Clean Sheet Factor", "concede2": "F1 Concede 2+ Goals Factor",
    "concede4": "F1 Concede 4+ Goals Factor", "saves3": "F1 3+ Saves Factor", "saves6": "F1 6+ Saves Factor",
}


def _future_win(m, season, opp, venue):
    """Reconciled own-team win / opp-win for one future fixture (per-player rows), like the F3-F8 loop."""
    home = venue == "H"
    o_t, o_r, o_6 = P._season_lookup(m, opp, season)
    w = model.win_pred(m["Title"], m["Relegation"], m["Top 6"], o_t, o_r, o_6, home).clip(0, 1)
    ow = model.opp_win_pred(m["Title"], m["Relegation"], o_t, o_r, home).clip(0, 1)
    uw, uo = P._unify_match(w, ow, m["Team"], opp)
    return model.reconcile_win_draw(uw, uo) + (home,)


def _xp_pre_future(m, w, ow, home):
    """XP-Pre for a future fixture: factor x baseline for each stat, DefCon held, then model.xp_pre."""
    pos = m["Position"]
    s = {}
    for key, fcol in STAT_FACTORS.items():
        s[key] = (m[fcol] * model.baseline(key, w, ow, pos, home)).clip(lower=0)
    s["score1"] = s["score1"].clip(upper=0.95)
    s["score2"], s["score3"] = model.poisson_score2(s["score1"]), model.poisson_score3(s["score1"])
    s["assist2"] = s["assist"] * config.ASSIST2_RATIO
    s["dc_def"] = m["F1 Defensive Contribution - DEF"]
    s["dc_mid"] = m["F1 Defensive Contribution - MID"]
    return model.xp_pre(pos, m["F8 Start"], s)


def build_matrix():
    m = pd.read_csv(MASTER)
    m["Position"] = m["Position"].astype(str)
    season = pd.read_csv(os.path.join(config.OUTPUTS_DIR, "04_season_probs.csv"))
    fx = pd.read_csv(os.path.join(config.INPUTS_DIR, "season_fixtures.csv"))

    # opponent + venue per team per GW (from the full-season list)
    opp_of, ven_of = {}, {}
    for gw in range(2, FIRST_HALF_END + 1):
        g = fx[fx["gameweek"] == gw]
        o, v = {}, {}
        for _, r in g.iterrows():
            o[r["home_team"]] = r["away_team"]; v[r["home_team"]] = "H"
            o[r["away_team"]] = r["home_team"]; v[r["away_team"]] = "A"
        opp_of[gw], ven_of[gw] = o, v

    # per-player XP-Pre + per-team win prob, GW2-19
    xp = pd.DataFrame({"Player Name": m["Player Name"], "Team": m["Team"], "Position": m["Position"],
                       "Cost": m["Cost"]})
    team_win = {}
    for gw in range(2, FIRST_HALF_END + 1):
        k = gw - 1                                           # GW -> F index
        if k <= 8:                                           # use the real pipeline projection
            xp[f"GW{gw}"] = m[f"F{k} XP Pre"]
            wcol = f"F{k} Win" if f"F{k} Win" in m else f"F{k} Win Pred"
            tw = m.groupby("Team")[wcol].first()
        else:                                                # extend with the same machinery
            opp = m["Team"].map(opp_of[gw]); ven = m["Team"].map(ven_of[gw])
            w, ow, home = _future_win(m, season, opp, ven)
            xp[f"GW{gw}"] = _xp_pre_future(m, w, ow, home)
            tw = pd.Series(w.values, index=m["Team"]).groupby(level=0).first()
        team_win[gw] = tw
    tw_mat = pd.DataFrame(team_win)                          # teams x GW win prob
    return xp, tw_mat, {gw: opp_of[gw] for gw in opp_of}, {gw: ven_of[gw] for gw in ven_of}


def _current_squad():
    gt = pd.read_csv(os.path.join(config.INPUTS_DIR, "gw_teams.csv"))
    i = gt.shape[1] - 1
    while i > 0 and gt.iloc[:, i].isna().all():
        i -= 1
    return gt.iloc[:, i].dropna().tolist()


def _transfer_targets(xp, tw, squad, opp_of, ven_of, lo, hi):
    """Best NON-OWNED players to buy into over the GW[lo..hi] window (e.g. ahead of a Wildcard)."""
    win = [f"GW{g}" for g in range(lo, hi + 1)]
    print(f"\n=== Transfer targets - best non-owned buys over GW{lo}-{hi} (fixture-ease window) ===")
    # team ease over the window, non-owned-context
    ease = (tw[list(range(lo, hi + 1))].mean(axis=1) * 100).sort_values(ascending=False)
    print("  Team ease (avg win% GW{}-{}):  ".format(lo, hi)
          + "  ".join(f"{t[:3]} {v:.0f}" for t, v in ease.head(8).items()))
    tgt = xp[~xp["Player Name"].isin(squad)].copy()
    tgt["win_xp"] = tgt[win].sum(axis=1)
    tgt["per_gw"] = tgt["win_xp"] / len(win)
    print(f"\n  Top non-owned by projected XP-Pre over the window (per-GW avg):")
    for posn in ["MID", "FWD", "DEF"]:
        top = tgt[tgt["Position"] == posn].sort_values("win_xp", ascending=False).head(6)
        print(f"  {posn}:")
        for _, r in top.iterrows():
            g0 = int(win[0][2:])
            fixt = f"{ven_of[g0].get(r['Team'],'?')}{str(opp_of[g0].get(r['Team'],'?'))[:3]}"
            print(f"    {r['Player Name'][:22]:<22} {r['Team'][:11]:<11} {r['Cost']:.1f}m  "
                  f"{r['per_gw']:.2f}/gw  (GW{g0}: {fixt})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-gw", type=int, default=8, dest="lo", help="transfer-target window start (WC GW)")
    ap.add_argument("--to-gw", type=int, default=15, dest="hi", help="transfer-target window end")
    args = ap.parse_args()
    xp, tw, opp_of, ven_of = build_matrix()
    gws = list(range(2, FIRST_HALF_END + 1))
    xp.to_csv(os.path.join(config.OUTPUTS_DIR, "fixture_horizon.csv"), index=False)

    # 1) FIXTURE MISMATCH - per-team win% matrix, ranked by first-half average
    tw = tw.reindex(sorted(tw.index, key=lambda t: -tw.loc[t].mean()))
    print("\n=== Fixture ease - team win probability, GW2-19 (F1/F2 real, F3-F8 pred, F9+ extended) ===")
    print("team           " + " ".join(f"{g:>3}" for g in gws) + "   avg")
    for t in tw.index:
        row = tw.loc[t]
        print(f"{t:<14} " + " ".join(f"{row[g] * 100:>3.0f}" for g in gws) + f"   {row.mean() * 100:>3.0f}")

    # 2) CURRENT SQUAD - owned teams' fixture ease, and fixture-driven upside per GW
    squad = _current_squad()
    own = xp[xp["Player Name"].isin(squad)].copy()
    matched = set(own["Player Name"])
    gwcols = [f"GW{g}" for g in gws]
    owned_teams = own["Team"].value_counts()
    print(f"\n=== Current squad ({len(matched)}/{len(squad)} matched) - owned teams' fixture ease (win%) ===")
    print("team (n)        " + " ".join(f"{g:>3}" for g in gws))
    for t in sorted(owned_teams.index, key=lambda t: -tw.loc[t].mean() if t in tw.index else 0):
        if t in tw.index:
            r = tw.loc[t]
            hard = " ".join(f"{r[g]*100:>3.0f}" for g in gws)
            print(f"{t[:11]:<11}({owned_teams[t]}) {hard}")

    # fixture-driven upside = XP-Pre minus the 2xstart appearance floor (isolates the fixture signal)
    m2 = pd.read_csv(MASTER).set_index("Player Name")
    floor = own["Player Name"].map(2.0 * m2["F8 Start"]).values
    up = own[gwcols].sub(pd.Series(floor, index=own.index), axis=0).clip(lower=0)
    per_gw = up.sum()
    print("\n  Squad fixture-driven upside per GW (XP above the appearance floor):")
    print("  " + "  ".join(f"GW{g}:{per_gw[f'GW{g}']:.0f}" for g in gws))
    worst = per_gw.sort_values().head(4)
    print("\n  Toughest weeks & the key starters with the hard fixture:")
    for gwc, val in worst.items():
        gw = int(gwc[2:])
        cand = own.assign(_u=up[gwc].values)
        cand = cand[cand["Cost"] >= 6.0].sort_values("_u").head(3)   # real assets, not bench fodder
        drag = ", ".join(f"{r['Player Name'].split()[-1]} {r['Team'][:3]}({ven_of[gw].get(r['Team'],'?')}{str(opp_of[gw].get(r['Team'],'?'))[:3]})"
                         for _, r in cand.iterrows())
        print(f"    {gwc} (upside {val:.0f}): {drag}")
    if len(matched) < len(squad):
        print("\n  unmatched squad names:", sorted(set(squad) - matched))

    # 3) TRANSFER TARGETS - non-owned buys over the window
    _transfer_targets(xp, tw, squad, opp_of, ven_of, args.lo, args.hi)


if __name__ == "__main__":
    main()
