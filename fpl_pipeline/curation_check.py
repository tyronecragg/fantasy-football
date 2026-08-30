"""Curation self-checks folded into every pipeline run.

They read the built master and flag lineup-belief problems the name-reconciliation pass does
not: a start probability above what FPL itself says is possible, and teams whose start
probabilities do not add up. Findings are appended to outputs/14_name_reconciliation.csv and
printed in the run summary. Nothing is auto-corrected — FPL availability flags can lag, and a
pending sale legitimately reads as "available", so these are surfaced for a human decision.
"""
import os

import pandas as pd

from . import config

_COLS = ["source", "name", "issue", "suggestion", "how"]

# Materiality thresholds — below these a breach is noise (an injured bench player graded 0.05
# is technically over a 0-ceiling but changes no decision), so they are counted, not listed.
CEIL_GAP = 0.10          # flag F1 start only when it exceeds FPL's ceiling by more than this
SUM_TOL = 0.20           # flag an F2-F8 team sum only when it dips this far below 11


def _fpl_ceiling():
    """player_id -> (status, ceiling) where ceiling is the max our F1 start prob may take given
    FPL's own availability flag. 'a' imposes no ceiling (1.0); 'd' caps at
    chance_of_playing_next_round/100 (0.75 if absent); 'i'/'u'/'s' force 0."""
    ps = pd.read_csv(os.path.join(config.FPL_DATA_DIR, "playerstats.csv"))
    if "gw" in ps.columns:                       # keep each player's latest row
        ps = ps.sort_values("gw").drop_duplicates("id", keep="last")

    def ceil(r):
        st = str(r.get("status", "a"))
        if st == "a":
            return 1.0
        c = r.get("chance_of_playing_next_round")
        if pd.notna(c):
            return float(c) / 100.0
        return 0.75 if st == "d" else 0.0

    ps = ps.copy()
    ps["_ceiling"] = ps.apply(ceil, axis=1)
    keep = ["status", "_ceiling"] + (["news"] if "news" in ps.columns else [])
    return ps.set_index("id")[keep]


def check(master):
    """Return a reconciliation-shaped DataFrame of curation violations (empty if all clean)."""
    rows = []

    # 1. Availability ceiling: F1 start prob above what FPL says is possible (material breaches only).
    ceil = _fpl_ceiling()
    m = master.merge(ceil, left_on="player_id", right_index=True, how="left")
    m["_ceiling"] = m["_ceiling"].fillna(1.0)
    breach = m[m["F1 Start"] > m["_ceiling"] + 1e-9]
    minor = int((breach["F1 Start"] - breach["_ceiling"] <= CEIL_GAP).sum())
    for _, r in breach[breach["F1 Start"] - breach["_ceiling"] > CEIL_GAP] \
            .sort_values("F1 Start", ascending=False).iterrows():
        news = str(r.get("news", "") or "")[:50]
        rows.append({"source": "curation:ceiling", "name": r["Player Name"],
                     "issue": f"F1 start {r['F1 Start']:.2f} > FPL ceiling {r['_ceiling']:.2f} "
                              f"(status {r.get('status', '?')}{'; ' + news if news else ''})",
                     "suggestion": f"{r['_ceiling']:.2f}", "how": "cap F1 to the ceiling"})
    if minor:
        rows.append({"source": "curation:ceiling", "name": f"({minor} minor)",
                     "issue": f"{minor} more F1 starts over a 0-ceiling by <= {CEIL_GAP:.2f} "
                              f"(injured/suspended fringe, immaterial)", "suggestion": "", "how": ""})

    # 2. Team start-prob sums: F1 must total 11 exactly; each of F2..F8 must total >= 11 (one row
    #    per team, reporting the worst-case sum so a shortfall shows once, not eight times).
    fk = [c for c in (f"F{k} Start" for k in range(1, 9)) if c in master.columns]
    for team, g in master.groupby("Team"):
        f1 = float(g["F1 Start"].sum())
        if abs(f1 - 11) > 0.05:
            rows.append({"source": "curation:team_sum", "name": team,
                         "issue": f"F1 Start sums to {f1:.2f} (must be 11 - roster/lineup mismatch?)",
                         "suggestion": "11.00", "how": ""})
        future = [float(g[c].sum()) for c in fk if c != "F1 Start"]
        if future and min(future) < 11 - SUM_TOL:
            rows.append({"source": "curation:team_sum", "name": team,
                         "issue": f"F2-F8 Start dips to {min(future):.2f} (must be >= 11 - "
                                  f"a lineup player missing from the roster?)", "suggestion": ">= 11", "how": ""})

    return pd.DataFrame(rows, columns=_COLS)


def print_summary(cur):
    if cur.empty:
        print("  curation checks: F1 sums to 11, F2-F8 >= 11, no availability-ceiling breaches")
        return
    print(f"  curation checks: {len(cur)} issue(s) -> outputs/14_name_reconciliation.csv")
    for _, r in cur[cur.source == "curation:ceiling"].iterrows():
        print(f"    [ceiling]  {r['name']}: {r['issue']}")
    for _, r in cur[cur.source == "curation:team_sum"].iterrows():
        print(f"    [team-sum] {r['name']}: {r['issue']}")
