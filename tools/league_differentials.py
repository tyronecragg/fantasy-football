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
import json
import os
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config  # noqa: E402

API = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0"}
DEFAULT_ENTRY = 1250175   # Tyrone
MASTER = os.path.join(config.OUTPUTS_DIR, "13_players_master.csv")

# --xp weighted-total weights. These MIRROR optimisation.py's OWNERSHIP_WEIGHTS / RELIABILITY_WEIGHTS
# (kept local because optimisation.py runs the optimiser at import — not import-safe). Fixture weight =
# ownership x reliability, the optimiser's own combine. Keep in sync if the optimiser's change.
OWNERSHIP_WEIGHTS = (1.0, 0.920, 0.846, 0.779, 0.716, 0.659, 0.606, 0.558)
RELIABILITY_WEIGHTS = (1.0, 0.850, 0.710, 0.630, 0.550, 0.470, 0.400, 0.330)   # points-measured 2026-09-02
# Bench weights are a display choice (Free-Hit-thin, so the weighted total barely credits subs).
XP_BENCH_SLOT_WEIGHTS = (0.25, 0.05, 0.01)   # 3 outfield bench slots by XP order
XP_GK_BENCH_WEIGHT = 0.001                    # bench GK, every fixture

# Local cache of the live-API pulls (manager squads per GW, league entry lists). Once a GW locks,
# picks never change, so we reuse the cache and skip ~1 API call per manager (which is what times out
# / rate-limits on big leagues). Pass --refresh to force a re-fetch. Cache lives in inputs/league_cache/.
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inputs", "league_cache")


def _cache_load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _cache_save(path, obj):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def _squads_path(gw):
    return os.path.join(CACHE_DIR, f"squads_gw{gw}.json")


def _entries_path(league):
    return os.path.join(CACHE_DIR, f"entries_{league}.json")


def _meta_path(gw):
    return os.path.join(CACHE_DIR, f"meta_gw{gw}.json")


CHIP_ABBR = {"wildcard": "WC", "bboost": "BB", "3xc": "TC", "freehit": "FH"}


def _get(session, path, retries=4):
    """GET the FPL API with retry+backoff — the API is flaky and times out intermittently, and a run
    makes 100+ calls (bootstrap + one per manager), so a single transient timeout must not kill it.
    Retries connection/timeout errors and 5xx; 4xx (e.g. bad league id) raises immediately."""
    last = None
    for i in range(retries):
        try:
            r = session.get(f"{API}/{path}", headers=HEADERS, timeout=30)
            if r.status_code >= 500:                            # transient server error -> retry
                last = requests.HTTPError(f"{r.status_code} for {path}")
            else:
                r.raise_for_status()                            # 4xx raises (bad id etc.)
                return r.json()
        except (requests.ConnectTimeout, requests.ReadTimeout, requests.ConnectionError) as e:
            last = e
        if i < retries - 1:
            time.sleep(1.5 * (i + 1))                           # 1.5s, 3s, 4.5s backoff
    raise last


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


def _free_transfers(current, chip_by_ev, upto_gw):
    """FT available for the deadline AFTER upto_gw, reconstructed from transfer history.

    FPL doesn't publish FT, but the rule is deterministic: GW1 accrues nothing (unlimited
    setup); each later normal GW accrues +1 (banked, capped 5) then spends event_transfers;
    Wildcard/Free Hit weeks accrue nothing and spend nothing (chip moves are free). The
    upcoming GW then adds its own +1 (assumes it isn't chipped)."""
    leftover = 0
    for ev in sorted(current, key=lambda r: r["event"]):
        g = ev["event"]
        if g > upto_gw:
            break
        is_transfer_chip = chip_by_ev.get(g) in ("wildcard", "freehit")
        added = 0 if (g == 1 or is_transfer_chip) else 1
        avail = min(5, leftover + added)
        used = 0 if is_transfer_chip else ev.get("event_transfers", 0)
        leftover = max(0, avail - used)
    return min(5, leftover + 1)                               # +1 for the upcoming GW


def _entry_meta(session, entry, upto_gw):
    """Per-manager meta from entry/{id}/ + entry/{id}/history/: team value & bank (£m),
    total transfers, chips used (abbrev + GW, alphabetical), and free transfers available
    for the deadline after upto_gw (reconstructed - see _free_transfers).

    total_transfers counts non-chip transfers only (chip-week moves don't count)."""
    e = _get(session, f"entry/{entry}/")
    h = _get(session, f"entry/{entry}/history/")
    value = e.get("last_deadline_value", 0) / 10.0            # squad value INCL bank (selling prices)
    bank = e.get("last_deadline_bank", 0) / 10.0
    total_tr = e.get("last_deadline_total_transfers", 0)
    chip_by_ev = {c["event"]: c["name"] for c in h.get("chips", [])}
    chips = " ".join(sorted(f"{CHIP_ABBR.get(c['name'], c['name'])}({c['event']})"
                            for c in h.get("chips", []))) or "-"
    ft = _free_transfers(h.get("current", []), chip_by_ev, upto_gw)
    return {"value": value, "bank": bank, "team": value - bank,
            "total_tr": total_tr, "chips": chips, "ft": ft}


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


_XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
_XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}


def _our_horizon(n=8):
    """Master rows indexed by FPL player_id: Position + F1..Fn XP (n capped to what the master has)."""
    m = pd.read_csv(MASTER)
    if "player_id" not in m.columns:
        return None, []
    fcols = [f"F{k} XP" for k in range(1, n + 1) if f"F{k} XP" in m.columns]
    m = m.dropna(subset=["player_id"]).copy()
    m["player_id"] = m["player_id"].astype(int)
    return m.set_index("player_id")[["Position"] + fcols], fcols


def _best_xi_pts(sub, col):
    """A squad's expected points for one fixture: best LEGAL XI (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD; 11)
    by `col`, plus the optimal captain (best starter doubled). NaN XP dropped."""
    d = sub[["Position", col]].dropna(subset=[col]).sort_values(col, ascending=False)
    chosen, cnt = [], {p: 0 for p in _XI_MAX}
    for p, need in _XI_MIN.items():                              # positional minimums (best each)
        for idx in d.index[d["Position"] == p][:need]:
            chosen.append(idx); cnt[p] += 1
    for idx in d.index:                                          # fill to 11, best remaining, respect max
        if len(chosen) >= 11:
            break
        if idx in chosen:
            continue
        p = d.at[idx, "Position"]
        if cnt.get(p, 0) < _XI_MAX.get(p, 0):
            chosen.append(idx); cnt[p] += 1
    vals = d.loc[chosen, col]
    return float(vals.sum() + (vals.max() if len(vals) else 0.0))


def _fixture_value(sub, col):
    """(best XI + captain, weighted bench) for one fixture. Bench = the non-XI players: top-3 outfield
    subs by `col` x XP_BENCH_SLOT_WEIGHTS + best bench GK x XP_GK_BENCH_WEIGHT (mirrors the optimiser's
    bench_points_for_fixture, so the weighted-with-bench total is comparable to what it optimises)."""
    d = sub[["Position", col]].dropna(subset=[col]).sort_values(col, ascending=False)
    chosen, cnt = [], {p: 0 for p in _XI_MAX}
    for p, need in _XI_MIN.items():
        for idx in d.index[d["Position"] == p][:need]:
            chosen.append(idx); cnt[p] += 1
    for idx in d.index:
        if len(chosen) >= 11:
            break
        if idx in chosen:
            continue
        p = d.at[idx, "Position"]
        if cnt.get(p, 0) < _XI_MAX.get(p, 0):
            chosen.append(idx); cnt[p] += 1
    xi = d.loc[chosen, col]
    xi_pts = float(xi.sum() + (xi.max() if len(xi) else 0.0))
    bench = d.drop(index=chosen)
    out = bench.loc[bench["Position"] != "GK", col].tolist()[:len(XP_BENCH_SLOT_WEIGHTS)]
    gk = bench.loc[bench["Position"] == "GK", col].tolist()[:1]
    bench_pts = sum(v * w for v, w in zip(out, XP_BENCH_SLOT_WEIGHTS)) + sum(v * XP_GK_BENCH_WEIGHT for v in gk)
    return xi_pts, float(bench_pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", type=int, help="classic league id (omit to list your leagues and exit)")
    ap.add_argument("--entry", type=int, default=DEFAULT_ENTRY, help="your entry id")
    ap.add_argument("--top", type=int, default=100, help="cap managers sampled by rank (one API call each)")
    ap.add_argument("--gw", type=int, help="gameweek for picks (default: current)")
    ap.add_argument("--min-league-own", type=float, default=25.0, help="template threshold (league own %)")
    ap.add_argument("--xp", action="store_true",
                    help="instead of differentials: tabulate each team's F1..Fn projected XP (best XI + "
                         "captain per fixture), sorted")
    ap.add_argument("--sort", choices=["f1", "total", "wtotal", "wbench"], default="total",
                    help="--xp sort key: f1 | total (raw) | wtotal (ownership x reliability weighted) | "
                         "wbench (weighted incl bench). Default total")
    ap.add_argument("--horizon", type=int, default=8, help="--xp: fixtures to show, F1..Fn (default 8)")
    ap.add_argument("--meta", action="store_true",
                    help="--xp: also fetch each manager's squad value, bank, total transfers, chips used "
                         "and inferred free transfers (one extra API call each; cached per GW)")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore the local cache and re-fetch squads + league entries from the API")
    args = ap.parse_args()

    s = requests.Session()
    if not args.league:                                          # no league -> list yours and stop
        team, leagues = _my_leagues(s, args.entry)
        _print_leagues(team, leagues)
        return

    cur, players = _bootstrap(s)
    gw = args.gw or cur

    # league entries: cached per league id (reused unless --refresh)
    ent_path = _entries_path(args.league)
    ent_cached = None if args.refresh else _cache_load(ent_path)
    if ent_cached and len(ent_cached.get("entries", [])) >= min(args.top, ent_cached.get("total", 0) or args.top):
        name = ent_cached["name"]
        entries = [tuple(e) for e in ent_cached["entries"]][:args.top]
        capped = ent_cached["capped"] or len(ent_cached["entries"]) > args.top
    else:
        name, entries, capped = _league_entries(s, args.league, args.top)
        _cache_save(ent_path, {"name": name, "entries": entries, "capped": capped, "total": len(entries)})

    # manager squads: cached per (gw, entry). One dict for the whole GW, reused across runs/leagues.
    sq_path = _squads_path(gw)
    sq_cache = {} if args.refresh else _cache_load(sq_path)
    _fetched = [0]

    def squad_of(entry):
        """Cached set of the manager's picked element ids (fetches + stores on a miss)."""
        k = str(entry)
        if k in sq_cache:
            return set(sq_cache[k])
        pids = _squad(s, entry, gw)                 # may raise requests.RequestException
        sq_cache[k] = sorted(pids)
        _fetched[0] += 1
        return pids

    # manager meta (value/bank/transfers/chips/FT): cached per GW, like squads
    meta_path = _meta_path(gw)
    meta_cache = {} if args.refresh else _cache_load(meta_path)
    _meta_fetched = [0]

    def meta_of(entry):
        """Cached per-manager meta dict (fetches + stores on a miss)."""
        k = str(entry)
        if k in meta_cache:
            return meta_cache[k]
        m = _entry_meta(s, entry, gw)               # may raise requests.RequestException
        meta_cache[k] = m
        _meta_fetched[0] += 1
        return m

    scope = (f"top {len(entries)} by rank (league is larger; raise --top for more)" if capped
             else f"all {len(entries)} managers")
    src = "cached" if (ent_cached and not args.refresh) else "live"
    print(f"\nLeague: {name}  |  {scope}  |  picks GW{gw}  |  entries: {src}\n")

    if args.xp:                                                  # tabulate each team's projected XP
        hz, fcols = _our_horizon(args.horizon)
        if hz is None or not fcols:
            print("no F XP columns in the master (need player_id + F1 XP..). Run the pipeline first.")
            return
        fw = [OWNERSHIP_WEIGHTS[k] * RELIABILITY_WEIGHTS[k] for k in range(len(fcols))]
        teams = []
        for eid, ename, rank in entries:
            try:
                sub = hz.reindex(list(squad_of(eid))).dropna(subset=["Position"])
            except requests.RequestException:
                continue
            xi_b = [_fixture_value(sub, c) for c in fcols]                    # [(xi+cap, bench), ...]
            per = [v[0] for v in xi_b]
            wtot = sum(fw[k] * per[k] for k in range(len(fcols)))             # ownership x reliability weighted
            wben = sum(fw[k] * (per[k] + xi_b[k][1]) for k in range(len(fcols)))  # + weighted bench
            meta = None
            if args.meta:
                try:
                    meta = meta_of(eid)
                except requests.RequestException:
                    meta = None
            teams.append((eid, ename, per, sum(per), wtot, wben, meta, rank))
        _cache_save(sq_path, sq_cache)
        if args.meta:
            _cache_save(meta_path, meta_cache)
        if _fetched[0] or _meta_fetched[0]:
            note = f"{_fetched[0]} squad(s)" + (f" + {_meta_fetched[0]} meta" if _meta_fetched[0] else "")
            print(f"  (fetched {note} from the API, cached under {os.path.relpath(CACHE_DIR)})\n")
        keyfn = {"f1": lambda t: -t[2][0], "total": lambda t: -t[3],
                 "wtotal": lambda t: -t[4], "wbench": lambda t: -t[5]}[args.sort]
        teams.sort(key=keyfn)
        hdr = "".join(f"{c.replace(' XP', ''):>7}" for c in fcols)
        sort_lbl = {"f1": "F1", "total": "TOTAL", "wtotal": "wTOT", "wbench": "wTOT+B"}[args.sort]
        meta_hdr = f"{'Value':>7}{'Bank':>6}{'FT':>4}  chips" if args.meta else ""
        print(f"Team projected XP (best XI + optimal captain per fixture), sorted by {sort_lbl}:\n")
        header = f"  {'#':>3}{'Pos':>5}  {'manager':<24}{hdr}{'TOTAL':>8}{'wTOT':>8}{'wTOT+B':>9}{meta_hdr}"
        rows = []                                                # (body-without-mark, mark) to size the rule
        for i, (eid, ename, per, tot, wtot, wben, m, rank) in enumerate(teams, 1):
            meta_str = ""
            if args.meta:
                meta_str = (f"{m['team']:>6.1f}m{m['bank']:>5.1f}m{m.get('ft', '-'):>4}  {m['chips']}"
                            if m else f"{'-':>7}{'-':>6}{'-':>4}  -")
            body = (f"  {i:>3}{rank:>5}  {ename[:24]:<24}" + "".join(f"{v:>7.1f}" for v in per)
                    + f"{tot:>8.1f}{wtot:>8.1f}{wben:>9.1f}{meta_str}")
            rows.append((body, "  <- you" if eid == args.entry else ""))
        width = max([len(header)] + [len(b) for b, _ in rows])   # rule spans the widest line (3-chip case)
        print(header)
        print("  " + "-" * (width - 2))
        for body, mark in rows:
            print(body + mark)
        return

    mine = squad_of(args.entry)
    counts = {}
    ok = 0
    for eid, ename, rank in entries:
        try:
            for pid in squad_of(eid):
                counts[pid] = counts.get(pid, 0) + 1
            ok += 1
        except requests.RequestException:
            continue
    _cache_save(sq_path, sq_cache)
    if _fetched[0]:
        print(f"  (fetched {_fetched[0]} squad(s) from the API, cached to {os.path.relpath(sq_path)})\n")
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
