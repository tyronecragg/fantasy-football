"""GW11 'Make the Call' — players who come on as a SUBSTITUTE earn TRIPLE points. One per club.

Our projections only count starts, so the sub route is added on top:

    P(sub on) = available × (1 − start chance) × sub-on rate when not starting
    extra     = 3 × P(sub on) × points per sub appearance

Sub-on rate and points per sub appearance come from FPL's per-gameweek data (build_subs.py):
this season plus last season at half weight, steadied toward the position average with PRIOR_N
pseudo-games so a couple of cameos can't dominate. 'available' is FPL's status flag (injured /
suspended / unavailable -> 0, doubtful -> its chance of playing), so an injured player isn't
credited with bench appearances just because he won't start.

Late swaps: list players confirmed ON THE BENCH in confirmed_bench — they stay in the pool with a
0% start chance (exactly the players this week rewards). confirmed_not_starting still means not in
the matchday squad at all (removed). The 'bring in if they start' / 'if a pick doesn't start'
sections are off this week: starting can be worth LESS than coming off the bench.
"""
import argparse
import glob
import os

import pandas as pd

import challenge_core as cc

SUBS_CSV = os.path.join(cc.INPUTS, "fpl_challenge_subs.csv")
CORE = os.path.join(cc.ROOT, "fpl_data", "FPL-Core-Insights", "data", "2026-2027", "By Gameweek")
SEASON, LAST_WEIGHT = "2026-2027", 0.5
PRIOR_N_RATE = 4.0      # pseudo non-starts at the position sub-on rate
PRIOR_N_PTS = 3.0       # pseudo sub appearances at the position points-per-appearance
SUB_MULT = 3.0


def availability():
    """player_id -> chance he is available this week, from FPL's latest status flag."""
    files = []
    for f in glob.glob(os.path.join(CORE, "GW*", "playerstats.csv")):
        try:
            gw = int(os.path.basename(os.path.dirname(f))[2:])
        except ValueError:
            continue
        files.append((gw, f))
    for gw, f in sorted(files, reverse=True):
        ps = pd.read_csv(f)
        if "status" not in ps.columns or ps["status"].isna().all():
            continue
        out = {}
        for _, r in ps.iterrows():
            st, ch = r.get("status"), r.get("chance_of_playing_next_round")
            if st == "a":
                out[r["id"]] = 1.0
            elif st == "d":
                out[r["id"]] = (float(ch) / 100.0) if pd.notna(ch) else 0.5
            else:                                   # injured / suspended / unavailable / not in league
                out[r["id"]] = 0.0
        return out
    return {}


def sub_profile(df):
    """Per-player (sub-on rate when not starting, points per sub appearance), aligned to df."""
    s = pd.read_csv(SUBS_CSV)
    pri = s[s["player_id"] == -1].set_index("position")
    s = s[s["player_id"] != -1].copy()
    s["w"] = s["season"].map(lambda x: 1.0 if x == SEASON else LAST_WEIGHT)
    s["nonstart"] = (s["gws"] - s["starts"]) * s["w"]
    s["apps_w"], s["pts_w"] = s["sub_apps"] * s["w"], s["sub_pts"] * s["w"]
    agg = s.groupby("player_id")[["nonstart", "apps_w", "pts_w"]].sum()

    rate, ppa = [], []
    for i in df.index:
        pos = df.loc[i, "Position"]
        pr = pri.loc[pos] if pos in pri.index else None
        p_rate = (pr["sub_apps"] / max(pr["gws"] - pr["starts"], 1)) if pr is not None else 0.25
        p_pts = (pr["sub_pts"] / max(pr["sub_apps"], 1)) if pr is not None else 1.3
        a = agg.loc[df.loc[i, "player_id"]] if df.loc[i, "player_id"] in agg.index else None
        n, apps, pts = (a["nonstart"], a["apps_w"], a["pts_w"]) if a is not None else (0.0, 0.0, 0.0)
        rate.append((apps + PRIOR_N_RATE * p_rate) / (n + PRIOR_N_RATE))
        ppa.append((pts + PRIOR_N_PTS * p_pts) / (apps + PRIOR_N_PTS))
    return pd.Series(rate, df.index), pd.Series(ppa, df.index)


def compute(df, confirmed_bench=()):
    """Adds p_sub, sub_rate, pts_per_sub, eff_xp, cap_bonus, boosted to df (in place)."""
    bench, missed = cc.match_names(df, list(confirmed_bench))
    if missed:
        cc._big_warning([f"confirmed_bench: NO MATCH for {', '.join(missed)} — not applied."])
    avail = df["player_id"].map(availability()).fillna(1.0)
    m = df["Player Name"].isin(bench)
    if m.any():
        df.loc[m, "F1 Start"] = 0.0            # on the bench: no starting points...
        df.loc[m, cc.XP_COL] = 0.0
        avail[m] = 1.0                         # ...but definitely in the squad
        print(f"  confirmed on the BENCH: {', '.join(sorted(bench))}")
    rate, ppa = sub_profile(df)
    start = df["F1 Start"].fillna(0.0)
    df["sub_rate"], df["pts_per_sub"] = rate, ppa
    df["p_sub"] = avail * (1.0 - start) * rate
    extra = SUB_MULT * df["p_sub"] * ppa
    df["eff_xp"] = df[cc.XP_COL] + extra
    df["cap_bonus"] = df["eff_xp"]
    df["boosted"] = extra > 0.5
    return df


def main(confirmed_not_starting=None, confirmed_starting=None, confirmed_bench=None,
         current_team=None, captain=None, now=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-club", type=int, default=1)
    args = ap.parse_args()

    df = cc.load_players(gw=11)   # snapshots/gw11.csv if it exists, else the live master
    df = cc.confirm_not_starting(df, list(confirmed_not_starting or []))   # not in the squad
    df = cc.confirm_starting(df, list(confirmed_starting or []))
    cc.set_live(df, current_team, captain, now)
    df = compute(df, confirmed_bench or [])

    title = ["FPL CHALLENGE  |  GW11 'Make the Call'  |  substitutes score TRIPLE",
             "extra = 3 × P(comes on) × points per sub appearance  (P = available × not starting × sub-on rate)"]
    cc.solve_and_report(df, args.max_per_club, title, "SUB x3",
                        show_watchlist=False, show_contingency=False)


if __name__ == "__main__":
    # confirmed_bench: named on the bench (the players this week rewards) — kept, start set to 0.
    # confirmed_not_starting: not in the matchday squad — removed. confirmed_starting: in the XI.
    # current_team / captain: once matches start, your six and captain for live locking.
    main(confirmed_not_starting=[], confirmed_starting=[], confirmed_bench=[],
         current_team=[], captain=None)
