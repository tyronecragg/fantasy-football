"""Total margin load on the GOALS and ASSISTS markets — the ones the pipeline consumes.

    python tools/margin_goals.py betway_matches.har

An earlier attempt (tools/margin_structure.py) measured the margin CURVE on shots on
target, then generalised to goals. That transfer was never justified: shots are not
consumed by the pipeline at all, and config.MARGIN_PLAYER = 1.05 is applied to goalscorer,
2+ goals, assists, bookings and saves. This measures the right markets.

IDENTIFICATION, and why it needs no assumed distribution. By linearity of expectation a
team's expected goals is the sum over its players of E[goals], and for each player
E[goals] = P(1+) + P(2+) + P(3+) + ... So:

    sum over the team's players of [P(1+) + P(2+)]  ==  that team's expected goals

The left side comes from the player markets. The right side comes from the CLEAN SHEET
market, which is a two-way yes/no pair and therefore de-marginable EXACTLY:
P(opponent keeps a clean sheet) = exp(-lambda_this_team), so lambda = -ln(P(CS)).

No Poisson assumption about individual players, no guess at how goals spread across a
squad, no convention constant. The ratio of the two sides IS the total load the bookmaker
has put on that market.

Assists reuse the same identity with one convention: not every goal is assisted, so
expected assists = lambda * ASSIST_RATE. That constant is stated, not hidden — and it only
scales the assist answer, leaving the goals answer untouched.
"""
import collections
import json
import math
import sys

import numpy as np

ASSIST_RATE = 0.70      # share of goals that carry an assist; the one convention here


def main(path="betway_matches.har"):
    sys.path.insert(0, ".")
    import tools.betway as bw
    import pandas as pd, os
    from fpl_pipeline import config
    roster = pd.read_csv(os.path.join(config.OUTPUTS_DIR, "01_fpl_players.csv"))
    team_of = dict(zip(roster["name"], roster["team"]))

    cs_csv = pd.read_csv(os.path.join(config.SPORTSBET_DIR, "sportsbet_clean_sheet_odds.csv"))
    har = json.load(open(path, encoding="utf-8"))
    ev = collections.defaultdict(lambda: collections.defaultdict(dict)); nm = {}
    for e in har["log"]["entries"]:
        u, b = e["request"]["url"], e["response"]["content"].get("text") or ""
        if "MarketGroupNamesAndMarketsForEvent" in u and b.startswith("{"):
            q = dict(p.split("=", 1) for p in u.split("?")[1].split("&") if "=" in p)
            for m, sel, sbv, odds in bw.selections(json.loads(b)):
                ev[q["eventId"]][m][(sel, sbv)] = odds
        elif "BetBook/Filtered" in u and b.startswith("{"):
            for x in json.loads(b).get("events", []):
                nm[str(x["eventId"])] = x.get("displayName") or x.get("name")

    gl, al = [], []
    print(f"{'fixture':<34}{'team':<16}{'lambda':>8}{'sum E[g]':>10}{'LOAD':>8}{'assists':>9}")
    for eid, bm in ev.items():
        # lambda per team from the clean-sheet pair (exact 2-way de-margining).
        # Sourced from the scraped CSV, not the HAR: this capture is player-markets only.
        fx = nm.get(eid, "")
        rows = cs_csv[cs_csv["match_name"] == fx]
        lam = {}
        for r in rows.itertuples():
            if pd.notna(r.clean_sheet_yes) and pd.notna(r.clean_sheet_no):
                p_cs = (1/r.clean_sheet_yes) / ((1/r.clean_sheet_yes) + (1/r.clean_sheet_no))
                lam[r.team_name] = -math.log(max(p_cs, 1e-6))   # goals the OPPONENT scores
        if len(lam) != 2:
            continue
        teams = list(lam)
        # E[goals] and E[assists] per team from the player markets
        eg = collections.defaultdict(float); ea = collections.defaultdict(float)
        for mkt, tgt in (("Anytime Goalscorer", eg), ("Player 1+ Assists", ea)):
            for (s, _), o in bm.get(mkt, {}).items():
                t = team_of.get(bw.player_name(s))
                if t in lam: tgt[t] += 1/o
        for mkt, tgt in (("Player Goals (Incl. Overtime)", eg),
                         ("Player Assists (Incl. Overtime)", ea)):
            for (s, _), o in bm.get(mkt, {}).items():
                n2, _, tail = s.rpartition(" ")
                if tail in ("2+", "3+"):                     # 1+ already counted above
                    t = team_of.get(bw.player_name(n2))
                    if t in lam: tgt[t] += 1/o
        for t in teams:
            opp = [x for x in teams if x != t][0]
            lam_t = lam[opp]           # opponent's clean sheet -> THIS team's goals
            if lam_t <= 0 or eg[t] == 0: continue
            load_g = eg[t] / lam_t
            load_a = ea[t] / (lam_t * ASSIST_RATE) if ea[t] else float("nan")
            gl.append(load_g); al.append(load_a)
            print(f"{nm.get(eid,eid)[:32]:<34}{t:<16}{lam_t:>8.2f}{eg[t]:>10.2f}"
                  f"{load_g:>8.2f}x{load_a:>8.2f}x")
    g = [x for x in gl if np.isfinite(x)]; a = [x for x in al if np.isfinite(x)]
    print()
    print(f"GOALS   mean load {np.mean(g):.2f}x   median {np.median(g):.2f}x   "
          f"range {min(g):.2f}-{max(g):.2f}   n={len(g)}")
    print(f"ASSISTS mean load {np.mean(a):.2f}x   median {np.median(a):.2f}x   "
          f"range {min(a):.2f}-{max(a):.2f}   n={len(a)}  (assumes {ASSIST_RATE:.0%} of goals assisted)")
    print(f"\nconfig.MARGIN_PLAYER = 1.05  ->  implies a {1.05:.2f}x load")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "betway_matches.har")
