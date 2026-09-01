"""Mini-league differentials: who your rivals own that you don't, and who you own that they don't.

Pulls a real FPL mini-league from the live API and computes INTRA-LEAGUE ownership (not global), then
joins it to our own XP projection (outputs/13_players_master.csv) so you see not just "12/50 rivals own
Haaland" but whether our model rates the chase. Two lists:
  - RIVALS' TEMPLATE YOU'RE MISSING : high league-ownership players you do NOT own (the threats)
  - YOUR DIFFERENTIALS              : your players with LOW league ownership (your edge)

    python tools/league_differentials.py                              # lists YOUR leagues + ids to pick from
    python tools/league_differentials.py --league 19669               # your entry defaults to Tyrone's
    python tools/league_differentials.py --league 301117 --top 30 --gw 1

With no --league it fetches your /entry/ and prints your classic leagues (id, name, size) so you never
have to hit the API by hand. --top caps how many managers (by league rank) to sample - for a big league
the top N are the rivals who matter and it keeps the run fast (one API call per manager)."""
import argparse
import os
import sys

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config  # noqa: E402

API = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0"}
DEFAULT_ENTRY = 1250175   # Tyrone
MASTER = os.path.join(config.OUTPUTS_DIR, "13_players_master.csv")


def _get(session, path):
    r = session.get(f"{API}/{path}", headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def _bootstrap(session):
    js = _get(session, "bootstrap-static/")
    cur = next((e["id"] for e in js["events"] if e.get("is_current")), None)
    pos = {t["id"]: t["singular_name_short"] for t in js["element_types"]}
    team = {t["id"]: t["short_name"] for t in js["teams"]}
    players = {e["id"]: {"name": e["web_name"], "pos": pos.get(e["element_type"], "?"),
                         "team": team.get(e["team"], "?"), "cost": e["now_cost"] / 10.0,
                         "global_own": float(e["selected_by_percent"])}
               for e in js["elements"]}
    return cur, players


def _squad(session, entry, gw):
    return {p["element"] for p in _get(session, f"entry/{entry}/event/{gw}/picks/")["picks"]}


def _league_entries(session, league, top):
    """Top `top` managers by league rank. Returns (name, entries, capped) — capped=True means the
    league has MORE managers than we took (we stopped at the cap), False means we got everyone."""
    ids, page, capped = [], 1, False
    while True:
        data = _get(session, f"leagues-classic/{league}/standings/?page_standings={page}")
        results = data["standings"]["results"]
        ids += [(r["entry"], r["entry_name"], r["rank"]) for r in results]
        has_next = data["standings"]["has_next"]
        if len(ids) >= top:
            capped = has_next or len(ids) > top
            break
        if not has_next:
            break
        page += 1
    return data["league"]["name"], ids[:top], capped


def _my_leagues(session, entry):
    """Your classic leagues from /entry/{id}/ — (id, name, size), so you don't fetch it by hand."""
    js = _get(session, f"entry/{entry}/")
    out = [(lg["id"], lg["name"], lg.get("rank_count") or 0) for lg in js["leagues"]["classic"]]
    return js.get("name", "?"), out


def _print_leagues(team, leagues):
    print(f"\nYour leagues ({team}) - pass one with --league <id> (small leagues = sharper differentials):\n")
    print(f"  {'id':>8}  {'size':>9}  name")
    for lid, nm, size in sorted(leagues, key=lambda x: x[2]):     # smallest (most personal) first
        print(f"  {lid:>8}  {size:>9,}  {nm[:48]}")
    print()


def _our_xp():
    if not os.path.exists(MASTER):
        return {}
    m = pd.read_csv(MASTER)
    if "player_id" not in m.columns:
        return {}
    f1 = "F1 XP" if "F1 XP" in m.columns else None
    tot = None  # Total XP retired; horizon value now comes from the optimiser, not the master
    return {int(r["player_id"]): (r.get(f1), r.get(tot)) for _, r in m.iterrows()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", type=int, help="classic league id (omit to list your leagues and exit)")
    ap.add_argument("--entry", type=int, default=DEFAULT_ENTRY, help="your entry id")
    ap.add_argument("--top", type=int, default=100, help="cap managers sampled by rank (one API call each)")
    ap.add_argument("--gw", type=int, help="gameweek for picks (default: current)")
    ap.add_argument("--min-league-own", type=float, default=25.0, help="template threshold (league own %)")
    args = ap.parse_args()

    s = requests.Session()
    if not args.league:                                          # no league -> list yours and stop
        team, leagues = _my_leagues(s, args.entry)
        _print_leagues(team, leagues)
        return

    cur, players = _bootstrap(s)
    gw = args.gw or cur
    name, entries, capped = _league_entries(s, args.league, args.top)
    scope = (f"top {len(entries)} by rank (league is larger; raise --top for more)" if capped
             else f"all {len(entries)} managers")
    print(f"\nLeague: {name}  |  {scope}  |  picks GW{gw}\n")

    mine = _squad(s, args.entry, gw)
    counts = {}
    ok = 0
    for eid, ename, rank in entries:
        try:
            for pid in _squad(s, eid, gw):
                counts[pid] = counts.get(pid, 0) + 1
            ok += 1
        except requests.HTTPError:
            continue
    n = max(ok, 1)
    xp = _our_xp()

    def row(pid):
        p = players.get(pid, {"name": f"id{pid}", "pos": "?", "team": "?", "cost": 0, "global_own": 0})
        lo = 100.0 * counts.get(pid, 0) / n
        f1, tot = xp.get(pid, (None, None))
        return (p["name"], p["pos"], p["team"], p["cost"], lo, p["global_own"], f1, tot)

    def show(rows, title):
        print(f"=== {title} ===")
        print(f"  {'player':<16}{'pos':<4}{'team':<5}{'£':>5}{'lg%':>6}{'own%':>6}{'F1xp':>6}{'TotXP':>7}")
        for nm, po, tm, c, lo, go, f1, tot in rows:
            f1s = f"{f1:.2f}" if pd.notna(f1) else " - "
            ts = f"{tot:.1f}" if pd.notna(tot) else " - "
            print(f"  {nm[:15]:<16}{po:<4}{tm:<5}{c:>5.1f}{lo:>5.0f}%{go:>5.0f}%{f1s:>6}{ts:>7}")
        print()

    # 1) rivals' template you're MISSING
    missing = sorted((row(pid) for pid in counts if pid not in mine and
                      100.0 * counts[pid] / n >= args.min_league_own), key=lambda r: -r[4])
    show(missing[:15], f"Rivals' template you're MISSING (>= {args.min_league_own:.0f}% of league, ranked by league own)")

    # 2) YOUR differentials (own, low league ownership)
    diffs = sorted((row(pid) for pid in mine), key=lambda r: r[4])
    show(diffs[:15], "YOUR players, least-owned in the league first (your differentials)")


if __name__ == "__main__":
    main()
