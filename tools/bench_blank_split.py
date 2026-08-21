# -*- coding: utf-8 -*-
"""Measure how often a NAILED starter blanks (0 minutes) the following gameweek, split by cause,
to ground the optimiser's bench_slot_weights.

    python tools/bench_blank_split.py [--season 2025-2026] [--nailed 4]

A starter-week = a player who started each of the last N PL matches (default 4) whose team has a
PL fixture next GW. A blank = he is absent from playermatchstats for that fixture (the file lists
every player who got minutes - coverage validated at 100% of known starters).

Each blank is classified with the FPL status snapshots in By Gameweek/GW*/playerstats.csv
(the GW k snapshot is the information state BEFORE the GW k+1 deadline):
  FORESEEABLE   flagged (status != 'a') in the GW k snapshot  -> the weekly re-run benches him
                and promotes a bench player at FULL weight; the bench weight is only the
                plan-time proxy for owning a playable bench body for these weeks.
  LATE INJURY   unflagged before, flagged in the GW k+1 snapshot -> the pure AUTOSUB event.
  ROTATION      never flagged -> rest/tactical; predicted-XI sources catch part of this, so for
                hard-1.0 declared starters the residual is lower than measured.
"back next GW" = he played again the GW after the blank (a one-week absence -> bench-cover
territory; multi-week absences are transfer territory).

2025-26 result (nailed = started last 4, outfield): blank 8.0%/starter-week = rotation 2.5% +
late injury 1.8% + foreseeable 3.8%. XI of 10: P(>=1 late injury) = 0.165 (this is the old
"0.16" autosub figure - it never included foreseeable absences), P(>=1 foreseeable) = 0.32 of
which ~39% are one-week.
"""
import argparse, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(season):
    D = os.path.join(ROOT, "fpl_data", "FPL-Core-Insights", "data", season, "By Gameweek")
    rec, tp, snap = [], {}, {}
    for k in range(1, 39):
        f = os.path.join(D, f"GW{k}")
        if not os.path.isdir(f):
            break
        m = pd.read_csv(os.path.join(f, "matches.csv"), usecols=["match_id", "tournament", "home_team", "away_team"])
        m = m[m.tournament == "prem"]
        pl = set(m.match_id)
        side = {}
        for _, r in m.iterrows():
            side[(r.match_id, "home")] = r.home_team
            side[(r.match_id, "away")] = r.away_team
        tp[k] = set(m.home_team.dropna()) | set(m.away_team.dropna())
        l = pd.read_csv(os.path.join(f, "lineups.csv"), usecols=["match_id", "team_side", "player_id", "position", "is_starting"])
        l = l[l.match_id.isin(pl) & l.player_id.notna()]
        pm = pd.read_csv(os.path.join(f, "playermatchstats.csv"), usecols=["player_id", "match_id", "minutes_played"])
        pm = pm[pm.match_id.isin(pl)]
        mins = pm.groupby("player_id").minutes_played.sum()
        snap[k] = pd.read_csv(os.path.join(f, "playerstats.csv"), usecols=["id", "status"]).set_index("id").status
        for _, r in l.iterrows():
            pid = int(r.player_id)
            rec.append((pid, k, side.get((r.match_id, r.team_side)), r.position, bool(r.is_starting), float(mins.get(pid, 0))))
    g = (pd.DataFrame(rec, columns=["pid", "gw", "team", "pos", "started", "mins"])
         .groupby(["pid", "gw"]).agg(team=("team", "first"), pos=("pos", "first"),
                                     started=("started", "max"), mins=("mins", "max")).reset_index())
    return g, tp, snap


def events(g, tp, snap, nailed):
    S = {(p, w): s for p, w, s in g[["pid", "gw", "started"]].itertuples(index=False)}
    M = {(p, w): m for p, w, m in g[["pid", "gw", "mins"]].itertuples(index=False)}
    T = {(p, w): t for p, w, t in g[["pid", "gw", "team"]].itertuples(index=False)}
    P = dict(zip(g.pid, g.pos))
    last = max(tp)
    ev = []
    for (p, k), s in S.items():
        if not s or k < nailed or k >= last:
            continue
        if not all(S.get((p, k - j), False) for j in range(1, nailed)):
            continue
        team = T[(p, k)]
        if team not in tp.get(k + 1, set()):
            continue
        blank = M.get((p, k + 1), 0.0) == 0.0
        pre = str(snap[k].get(p, "a")) != "a"
        post = str(snap.get(k + 1, pd.Series(dtype=object)).get(p, "a")) != "a"
        back = ((team in tp.get(k + 2, set())) and M.get((p, k + 2), 0.0) > 0) if k + 2 <= last else np.nan
        ev.append((P[p], blank, pre, post, back))
    return pd.DataFrame(ev, columns=["pos", "blank", "pre", "post", "back"])


def report(e, label, slots=10):
    n = len(e)
    b = e[e.blank]
    buckets = [("rotation/rest (never flagged)", b[~b.pre & ~b.post]),
               ("late injury/ban (flagged only after)", b[~b.pre & b.post]),
               ("foreseeable (flagged pre-deadline)", b[b.pre])]
    print(f"\n{label}: starter-weeks={n}  blanks={len(b)} ({len(b)/n:.2%})")
    for name, sub in buckets:
        p = len(sub) / n
        if slots == 1:
            print(f"  {name:38} {p:.2%}/week   back next GW {sub.back.mean():.0%}")
        else:
            p1 = 1 - (1 - p) ** slots
            p2 = p1 - slots * p * (1 - p) ** (slots - 1)
            print(f"  {name:38} {p:.2%}/starter-week  back next GW {sub.back.mean():.0%} | XI of {slots}: P(>=1)={p1:.3f} P(>=2)={p2:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2025-2026")
    ap.add_argument("--nailed", type=int, default=4, help="started each of the last N PL matches")
    a = ap.parse_args()
    g, tp, snap = load(a.season)
    e = events(g, tp, snap, a.nailed)
    report(e[e.pos != "G"], f"OUTFIELD (nailed = started last {a.nailed})", slots=10)
    report(e[e.pos == "G"], f"GOALKEEPER (nailed = started last {a.nailed})", slots=1)
