"""Scrape Betway South Africa odds into the CSV shapes the pipeline already reads.

    python tools/betway.py                  # discover fixtures, fetch every market
    python tools/betway.py --har betway.har # parse a saved capture instead
    python tools/betway.py --dry-run        # show what WOULD be written

Why Betway rather than Sportsbet or bet365:
  * plain `requests` works — no VPN, no TLS impersonation, no browser, no session token
    (bet365 gates its content APIs behind a WebSocket-bound token; Sportsbet 403s from
    South Africa without a VPN)
  * proper JSON with priceDecimal already computed, rather than a delimited wire format
  * it carries PLAYER 1+ ASSISTS, which Sportsbet never did — the market the projection
    machinery most wanted and previously had to synthesise

PARTIAL WRITES ARE THE POINT. Bookmakers publish player props progressively: goalkeeper
saves typically appear near kickoff. This writes ONLY the markets it actually finds and
leaves the rest alone, so the market-anchored placeholders from
tools/build_preseason_data.py survive untouched for anything not yet priced. Rerun closer
to kickoff and the real markets replace the estimates one at a time.

TWO GAMEWEEKS. The fixtures come back kickoff-ordered; the first ten are the current
gameweek (F1), the next ten are the following one (F2) - the same positional split the
pipeline makes on the win/draw/win file (team_model: iloc[:10]=F1, iloc[10:20]=F2). F1
fills the main sportsbet_*.csv files; F2's win odds extend the win/draw/win file and its
clean sheets / team goals fill the *_f2 files. Player props are scraped for F1 only - the
pipeline model-projects F2's from the F2 win odds - so early in the season, when only GW1
is priced, F2 is simply absent and those markets stay model-projected.
"""
import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config, names  # noqa: E402

BASE = "https://www.betway.co.za/sportsapi/br/v1"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/131.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.betway.co.za/",
}
COMMON = {"countryCode": "ZA", "cultureCode": "en-US"}

# Betway market displayName -> the pipeline file it feeds. Only these are written; any
# market absent from the response simply leaves its existing file (real or placeholder).
MARKETS = {
    "Anytime Goalscorer": "score1",
    "Player 1+ Assists": "assist",
    "Player To Be Booked": "yellow",
    "Total Goals": "team_goals",
}
# Ladder markets carry the threshold as a SUFFIX ON THE SELECTION ("Magalhaes, Gabriel 2+"),
# not in sbv — which is empty for these. This is where 2+ goals and 2+ assists live; there
# is no "Player 2+ Goals" market despite the 1+ one being named that way.
LADDERS = {
    "Player Goals (Incl. Overtime)": {1: "score1", 2: "score2"},
    "Player Assists (Incl. Overtime)": {1: "assist", 2: "assist2"},
}
SAVES_HINTS = ("saves", "goalkeeper saves")     # naming unknown until they publish it
# Shots-on-target ladders: P(1+)..P(5+) sum to E[shots on target] per player, which is the
# raw material for a saves estimate (saves ~ opposition SoT - goals conceded). Captured
# whenever present so the derivation can be built without another scrape.
SOT_MARKETS = tuple(f"Player {n}+ Shots On Target" for n in range(1, 6))
# The SAME fixture can carry shots-on-target in two shapes at once, and they are
# COMPLEMENTARY, not duplicates: on 2026-08-19 one fixture had 42 players in the fixed
# markets and 36 in the ladder, with 5 players only in the ladder and 11 only in the fixed
# set — 47 between them. The ladder also reaches 5+ where the fixed markets stop at 4+, so
# it captures more of the tail that E[shots on target] depends on.
SOT_LADDER = "Player Shots On Goal (Incl. Overtime)"
# NOT the same stat — "Player Shots (Incl. Overtime)" counts ALL shots (out to 8+), not
# shots on target. Feeding it to the saves model would inflate every keeper.
NOT_SOT = "Player Shots (Incl. Overtime)"


def player_name(raw):
    """Betway writes players as 'Surname, Firstname' — flip to our 'Firstname Surname'.

    Also repairs their casing: they lower-case inside names, so 'Mcburnie' and 'O'brien'
    come back where FPL has 'McBurnie' and 'O'Brien'. Everything left over goes through
    the shared name_mappings table like any other source.
    """
    raw = str(raw).strip()
    if "," in raw:
        surname, _, first = raw.partition(",")
        raw = f"{first.strip()} {surname.strip()}".strip()
    parts = []
    for word in raw.split():
        if word[:2].lower() == "mc" and len(word) > 2:
            word = "Mc" + word[2].upper() + word[3:]
        elif word[:2].lower() == "o'" and len(word) > 2:
            word = "O'" + word[2].upper() + word[3:]
        parts.append(word)
    return names.apply_player_names(pd.Series([" ".join(parts)]))[0]


def get(path, **params):
    r = requests.get(f"{BASE}/{path}", params={**COMMON, **params}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json() if r.content else {}


def fixtures(limit=None, region="england", league="premier-league"):
    """Fixtures AND their 1X2 prices in one call, ordered by kickoff.

    The league filter is required — without RegionAndLeagueIds the endpoint returns an
    empty payload rather than everything. Returns (rows, wdw): rows are
    (eventId, 'Home vs. Away', iso_date) sorted earliest-first; wdw is the win/draw/win
    frame with a leading 'match' column so it can be re-ordered into F1/F2 blocks later
    (assign_f1_f2 splits on kickoff order — the pipeline reads the file positionally).
    """
    data = get("BetBook/Filtered/", sportId="soccer", Skip=0, Take=60,
               isEsport="false", boostedOnly="false", SortOrder="League",
               marketTypes="[Win/Draw/Win]",
               **{"RegionAndLeagueIds[0].regionId": region,
                  "RegionAndLeagueIds[0].leagueId": league})

    price = {p["outcomeId"]: p.get("priceDecimal") for p in data.get("prices", [])}
    by_event = defaultdict(dict)
    for o in data.get("outcomes", []):
        odds = price.get(o["outcomeId"])
        if odds:
            # Key by the CANONICAL team name: split_teams() returns mapped names ('Man Utd'),
            # while Betway's outcome displayName is its own form ('Manchester United'), so an
            # unmapped key silently dropped the price for every team needing a rename — six of
            # ten GW1 2026-27 fixtures lost a win price this way (fixed 2026-08-21).
            raw = (o.get("displayName") or "").strip()
            key = raw if raw.lower() == "draw" else names.apply_team_names(pd.Series([raw]))[0]
            by_event[o["eventId"]][str(key).strip().lower()] = odds

    evs = []
    for ev in data.get("events", []):
        if ev.get("isFinished") or not ev.get("shouldDisplay", True):
            continue
        dt = pd.to_datetime(ev.get("expectedStartEpoch"), unit="s", errors="coerce")
        evs.append((dt, ev["eventId"], ev.get("displayName") or ev.get("name") or ""))
    evs.sort(key=lambda e: (pd.isna(e[0]), e[0]))          # kickoff order, undated last

    rows, wdw = [], []
    for dt, eid, label in evs:
        rows.append((eid, label, dt.isoformat() if pd.notna(dt) else ""))
        home, away = split_teams(label)
        prices = by_event.get(eid, {})
        if home and prices:
            wdw.append({"match": label, "home_team": home, "away_team": away,
                        "home_win_odds": prices.get(home.lower()),
                        "away_win_odds": prices.get(away.lower()),
                        "draw_odds": prices.get("draw")})
    if limit:
        rows = rows[:limit]
    return rows, pd.DataFrame(wdw)


PER_GW = 10        # a Premier League gameweek is ten fixtures


def assign_f1_f2(rows, per_gw=PER_GW):
    """Split the kickoff-ordered fixtures into the current gameweek (F1, first `per_gw`)
    and the next one (F2, the following `per_gw`) — exactly how the pipeline reads them:
    team_model.team_fixture_view takes wdw.iloc[:10] as F1 and wdw.iloc[10:20] as F2.

    Player props are scraped for F1 only; the pipeline model-projects F2's score/assist
    from the F2 win odds, so scraping them would be wasted (and would corrupt the F1
    player files, which are keyed by player+fixture, not gameweek).

    Returns (gw_of, ordered_rows): gw_of maps each match label to 'f1'/'f2'; ordered_rows
    is the F1 block then the F2 block. Anything past two gameweeks is dropped — the
    pipeline never reads scraped odds beyond F2.
    """
    f1, f2 = rows[:per_gw], rows[per_gw:2 * per_gw]
    gw_of = {label: "f1" for _, label, _ in f1}
    gw_of.update({label: "f2" for _, label, _ in f2})

    def span(chunk):
        ds = [d[:10] for *_, d in chunk if d]
        return f"{ds[0]}..{ds[-1]}" if ds else "no dates"

    print(f"  current GW (F1): {len(f1)} fixtures  [{span(f1)}]")
    if f2:
        print(f"  next GW    (F2): {len(f2)} fixtures  [{span(f2)}]")
    else:
        print("  next GW    (F2): not listed yet — F2 team markets stay model-projected")
    for tag, chunk in (("F1", f1), ("F2", f2)):
        if chunk and len(chunk) != per_gw:
            print(f"  WARNING: {tag} has {len(chunk)} fixtures, not {per_gw}. The pipeline "
                  "splits the win/draw/win file positionally (first 10 = F1, next 10 = F2), "
                  "so eyeball the [F1]/[F2] tags below to be sure the gameweeks line up.")
    return gw_of, f1 + f2


def _scheduled_pairings(gw_col):
    """{frozenset(home, away)} for a gameweek column of inputs/fixtures.csv (the pipeline's
    own schedule), or None if the file/column is missing. Columns are labelled by ABSOLUTE
    gameweek (build_fixtures --gw N writes 'GW{N} Opponent' first), so callers pass the actual
    first/second Opponent column — this is the authority the kickoff-order split is checked against."""
    path = os.path.join(config.ROOT, "inputs", "fixtures.csv")
    if not os.path.exists(path):
        return None
    fx = pd.read_csv(path)
    if gw_col not in fx.columns:
        return None
    team_col = fx.columns[0]
    pairings = set()
    for _, r in fx.iterrows():
        a = names.apply_team_names(pd.Series([str(r[team_col])]))[0]
        b = names.apply_team_names(pd.Series([str(r[gw_col])]))[0]
        if a and b and str(b).lower() != "nan":
            pairings.add(frozenset((a, b)))
    return pairings or None


def crosscheck_split(f1, f2):
    """Warn if the kickoff-order F1/F2 split disagrees with inputs/fixtures.csv's current /
    next gameweek pairings — the rare case a postponed fixture scrambles date order and puts
    a wrong-gameweek match in a block the pipeline reads positionally. Advisory only."""
    # The window's columns are labelled by ABSOLUTE gameweek (build_fixtures --gw N writes
    # 'GW{N} Opponent' first), so take the first two Opponent columns rather than hardcoding GW1/GW2.
    fpath = os.path.join(config.ROOT, "inputs", "fixtures.csv")
    opp_cols = ([c for c in pd.read_csv(fpath, nrows=0).columns if c.endswith("Opponent")]
                if os.path.exists(fpath) else [])
    opp_cols += [None, None]
    for tag, chunk, col in (("F1", f1, opp_cols[0]), ("F2", f2, opp_cols[1])):
        if col is None:
            continue
        sched = _scheduled_pairings(col)
        if not sched or not chunk:
            continue
        got = {frozenset(split_teams(m)) for _, m, _ in chunk if all(split_teams(m))}
        stray = [p for p in got if p not in sched]
        if stray:
            print(f"  WARNING: {tag} block has fixtures not in fixtures.csv '{col}': "
                  + ", ".join(" v ".join(sorted(p)) for p in stray)
                  + " — kickoff order may be scrambled (postponement?); verify the split.")


def split_teams(label):
    """'Arsenal vs. Coventry City' -> canonical ('Arsenal', 'Coventry City')."""
    for sep in (" vs. ", " vs ", " v "):
        if sep in label:
            a, b = label.split(sep, 1)
            return (names.apply_team_names(pd.Series([a.strip()]))[0],
                    names.apply_team_names(pd.Series([b.strip()]))[0])
    return None, None


GROUPS = ("Player", "Goals", "Team", "Main")   # the ones carrying FPL-relevant markets


def markets_for(event_id, take=250, groups=GROUPS, delay=0.6):
    """Fetch each market group separately and merge.

    A blank marketGroupId returns a TRUNCATED view — 134 markets when group-names reports
    ~380 across the groups, and 'Player Specials' alone holds 101. Player props (assists,
    2+ goals, shots on target, saves) live in that group, so asking for it by name is the
    difference between seeing them and silently missing them.
    """
    merged = {"marketsInGroup": [], "outcomes": [], "prices": [], "marketGroupNames": []}
    seen_markets, seen_outcomes = set(), set()
    for gid in groups:
        try:
            payload = get("MarketGroupings/MarketGroupNamesAndMarketsForEvent",
                          eventId=event_id, marketGroupId=gid, skip=0, take=take,
                          isBuildABetOnly="false")
        except Exception:
            continue                                  # group absent for this fixture
        for m in payload.get("marketsInGroup", []):
            if m["marketId"] not in seen_markets:
                seen_markets.add(m["marketId"])
                merged["marketsInGroup"].append(m)
        for o in payload.get("outcomes", []):
            if o["outcomeId"] not in seen_outcomes:
                seen_outcomes.add(o["outcomeId"])
                merged["outcomes"].append(o)
        merged["prices"] += payload.get("prices", [])
        time.sleep(delay)
    return merged


def selections(payload):
    """Flatten their relational payload to [(market, selection, sbv, decimal odds)]."""
    market_name = {m["marketId"]: (m.get("displayName") or m.get("name") or "")
                   for m in payload.get("marketsInGroup", [])}
    price = {p["outcomeId"]: p.get("priceDecimal") for p in payload.get("prices", [])}
    rows = []
    for o in payload.get("outcomes", []):
        odds = price.get(o["outcomeId"])
        if odds:
            rows.append((market_name.get(o["marketId"], ""),
                         o.get("displayName") or o.get("name") or "",
                         o.get("sbv") or "", float(odds)))
    return rows


def _pool_scale(pairs_by_match, min_pairs=8):
    """Pooled ladder->fixed rescale from EVERY fixture's players-priced-in-both.

    pairs_by_match maps match -> [(1/fixed_odds, 1/ladder_odds), ...]. The ladder's extra
    margin is a property of the market, not the match, so we pool all fixtures into one factor
    sum(1/fixed)/sum(1/ladder) rather than measuring one per fixture: steadier, and it covers
    fixtures that individually have too few players to trust. Returns (pooled, n_players,
    fixture_lo, fixture_hi) - the per-fixture range is a diagnostic so a scrape can show the
    factor barely moves - or None when fewer than min_pairs players match across all fixtures.
    """
    allp = [p for ps in pairs_by_match.values() for p in ps]
    if len(allp) < min_pairs:
        return None
    pooled = sum(f for f, _ in allp) / sum(l for _, l in allp)
    per = [sum(f for f, _ in ps) / sum(l for _, l in ps)
           for ps in pairs_by_match.values() if len(ps) >= 3]
    lo, hi = (min(per), max(per)) if per else (pooled, pooled)
    return pooled, len(allp), lo, hi


def collect(fixture_rows, gw_of, delay=3.0):
    """{pipeline key: DataFrame} for whichever markets Betway is currently pricing.

    Routes by gameweek (gw_of maps each match to 'f1'/'f2'): player props, saves and
    shots-on-target are scraped for the current gameweek only — the pipeline model-projects
    the next gameweek's from the F2 win odds. Team markets (clean sheets, team goals) are
    captured for both and land in the F1 files or the *_f2 files.

    `delay` seconds between fixtures. Each F1 call pulls ~800KB; F2 fetches only the lighter
    team groups. This runs weekly at most — be a good guest on someone else's API.
    """
    player_rows = defaultdict(list)      # score1 / score2 / assist / assist2 / yellow
    team_goals, clean_sheet, saves, sot = [], [], [], []
    team_goals_f2, clean_sheet_f2 = [], []
    ladder_scales = []
    ladder_raw = defaultdict(list)   # goals/assists ladder rungs (match, player, level, odds)
    sot_ladder_raw = []              # shots-on-target ladder rungs, likewise
    seen_markets = set()

    for event_id, match, date in fixture_rows:
        gw = gw_of.get(match, "f1")
        home, away = split_teams(match)
        # F2 needs only team-level markets (1X2 came with the fixture list; F2 player props
        # are model-projected) — fetch the lighter groups and skip the player pass below.
        groups = GROUPS if gw == "f1" else ("Team", "Goals", "Main")
        rows = selections(markets_for(event_id, groups=groups))
        seen_markets.update(m for m, *_ in rows)
        by_market = defaultdict(list)
        for market, sel, sbv, odds in rows:
            by_market[market].append((sel, sbv, odds))

        # per-team clean sheets: "<Team> To Keep A Clean Sheet" (both gameweeks). Match the
        # market's OWN team name (mapped to the roster name) against the fixture, rather than
        # the fixture team's first word against the market string: Betway's official names
        # ("Tottenham Hotspur", "Nottingham Forest") share no first word with the roster names
        # ("Spurs", "Nott'm Forest"), which silently dropped exactly those two clean sheets.
        cs_bucket = clean_sheet if gw == "f1" else clean_sheet_f2
        for k in by_market:
            kl = k.lower()
            if "clean sheet" not in kl or "1st half" in kl or " to keep" not in kl:
                continue
            mteam = names.apply_team_names(pd.Series([k[:kl.index(" to keep")].strip()]))[0]
            if mteam in (home, away):
                p = {s.lower(): o for s, _, o in by_market[k]}
                cs_bucket.append({"match_name": match, "date": date, "team_name": mteam,
                                  "clean_sheet_yes": p.get("yes"), "clean_sheet_no": p.get("no")})

        # Per-team goal totals -> the team-goals shape the pipeline reads (both gameweeks).
        # Betway prices each side separately as "<Team> Total (X.5)" (Over/Under). We read those
        # rather than the whole-match "Total Goals" (which can't distinguish the two sides — it
        # gave every team the same ~75% concede). The line comes from the market NAME; the line's
        # sbv and the 'Over '/'Under ' selection names both carry stray spaces, so strip them.
        # concede_market() reads the file POSITIONALLY (cols 4-5 = the 1.5/2+ line, 6-7 = 3.5/4+)
        # and keys on the Opponent column, so emit both perspectives with each team's OWN totals.
        team_tot = defaultdict(dict)
        for market, mrows in by_market.items():
            mm = re.match(r"^(.+?) Total \((\d\.5)\)$", market)
            if not mm:
                continue
            tname = names.apply_team_names(pd.Series([mm.group(1).strip()]))[0]
            for sel, _, odds in mrows:
                team_tot[(tname, mm.group(2))][sel.strip().lower()] = odds
        if any(t in (home, away) for t, _ in team_tot):
            tg_bucket = team_goals if gw == "f1" else team_goals_f2
            for a, b in ((home, away), (away, home)):
                row = {"Match": match, "Date": date, "Team": a, "Opponent": b}
                # Team_Over/Under for BOTH lines first (concede_market reads cols 4-5 = 1.5,
                # 6-7 = 3.5 positionally), then the mirrored Opponent_Concedes columns.
                for want in ("1.5", "3.5"):
                    o = team_tot.get((a, want), {})
                    row[f"Team_Over_{want}"] = o.get("over")
                    row[f"Team_Under_{want}"] = o.get("under")
                for want in ("1.5", "3.5"):
                    o = team_tot.get((a, want), {})
                    row[f"Opponent_Concedes_Over_{want}"] = o.get("over")
                    row[f"Opponent_Concedes_Under_{want}"] = o.get("under")
                tg_bucket.append(row)

        if gw != "f1":
            print(f"  [F2] {match:<32} {len(rows):>5} selections, {len(by_market):>3} markets")
            time.sleep(delay)
            continue

        # player markets (1X2 already came back with the fixture list)
        for market, key in MARKETS.items():
            if key == "team_goals":
                continue
            for sel, _, odds in by_market.get(market, []):
                player_rows[key].append({"player_name": player_name(sel),
                                         "match_id": match, "odds_decimal": odds})

        # ladder markets: "<Surname, Firstname> 2+" -> threshold from the selection suffix.
        # LADDERS CARRY MORE MARGIN THAN THE FIXED MARKETS (summed implied probability runs
        # ~3.7x true for ladders vs ~3.0x for fixed, on both goals and assists). Left as-is a
        # player gap-filled from a ladder gets a factor ~28% higher than an identical player
        # priced from the fixed market — a RELATIVE distortion that does not cancel. Stash the
        # rungs now; they are rescaled to the fixed level POOLED across every fixture after the
        # loop (one factor per market+rung is steadier than a per-fixture one, and covers thin
        # fixtures the old per-fixture >=8 rule would have skipped and left un-rescaled).
        for market in LADDERS:
            for sel, _, odds in by_market.get(market, []):
                nm, _, tail = sel.rpartition(" ")
                if tail.endswith("+") and tail[:-1].isdigit():
                    ladder_raw[market].append((match, player_name(nm), int(tail[:-1]), odds))

        # goalkeeper saves, if they have published it yet
        for market in by_market:
            if any(h in market.lower() for h in SAVES_HINTS):
                for sel, sbv, odds in by_market[market]:
                    saves.append({"Match": match, "Date": date, "Team": "",
                                  "Goalkeeper": sel, "market": market, "odds": odds})

        # shots-on-target ladders — raw material for a saves estimate if the saves market
        # never appears (saves ~ opposition SoT - goals conceded)
        seen_sot = set()          # (player, threshold) — never take a level twice
        for market in SOT_MARKETS:
            n = int(market.split()[1].rstrip("+"))
            for sel, _, odds in by_market.get(market, []):
                who = player_name(sel)
                if (who, n) in seen_sot:
                    continue
                seen_sot.add((who, n))
                sot.append({"Match": match, "Date": date, "player_name": who,
                            "threshold": n, "odds_decimal": odds})
        # shots-on-target ladder: stash for the same pooled rescale as goals/assists. Its 5+
        # rung exists ONLY in the ladder (no fixed market to anchor against), so after the loop
        # it inherits the mean of the rungs that could be measured.
        for sel, _, odds in by_market.get(SOT_LADDER, []):
            nm2, _, tail = sel.rpartition(" ")
            if tail.endswith("+") and tail[:-1].isdigit():
                sot_ladder_raw.append((match, date, player_name(nm2), int(tail[:-1]), odds))

        print(f"  [F1] {match:<32} {len(rows):>5} selections, {len(by_market):>3} markets")
        time.sleep(delay)

    # ---- pooled ladder rescaling: one factor per market+rung across EVERY fixture ----
    # Goals/assists first. The fixed anchor (player_rows[key]) holds only the fixed markets at
    # this point, since ladder application was deferred. The 2+ rungs have no fixed market to
    # anchor against, so they INHERIT the 1+ rung's factor: a ladder's markup is ~uniform across
    # its rungs, so the measured 1+ shrink is a far better estimate for 2+ than leaving the full
    # ladder margin on (which was quietly under-stating 2+ goals and 2+ assists).
    for market, levels in LADDERS.items():
        scale, report = {}, {}
        for lvl, key in levels.items():
            fixed_px = {(p["match_id"], p["player_name"]): p["odds_decimal"]
                        for p in player_rows.get(key, [])}
            by_match = defaultdict(list)
            for mt, who, l2, odds in ladder_raw[market]:
                if l2 == lvl and (mt, who) in fixed_px:
                    by_match[mt].append((1.0 / fixed_px[(mt, who)], 1.0 / odds))
            pooled = _pool_scale(by_match)
            if pooled:
                scale[lvl], report[lvl] = pooled[0], pooled
        anchor = scale.get(1, 1.0)     # 1+ factor; the 2+ rung (no fixed market) inherits it
        for mt, who, lvl, odds in ladder_raw[market]:
            key = levels.get(lvl)
            if key:
                player_rows[key].append({"player_name": who, "match_id": mt,
                                         "odds_decimal": round(odds / scale.get(lvl, anchor), 2)})
        if report:
            ladder_scales.append((market.split()[1], report))

    # shots-on-target: same pooled rescale; the ladder-only 5+ rung inherits the mean factor
    fixed_sot = {(r["Match"], r["player_name"], r["threshold"]): r["odds_decimal"] for r in sot}
    sot_scale, sot_report = {}, {}
    for lvl in sorted({l for _, _, _, l, _ in sot_ladder_raw}):
        by_match = defaultdict(list)
        for mt, dt, who, l2, odds in sot_ladder_raw:
            if l2 == lvl and (mt, who, lvl) in fixed_sot:
                by_match[mt].append((1.0 / fixed_sot[(mt, who, lvl)], 1.0 / odds))
        pooled = _pool_scale(by_match)
        if pooled:
            sot_scale[lvl], sot_report[lvl] = pooled[0], pooled
    default_scale = (sum(sot_scale.values()) / len(sot_scale)) if sot_scale else 1.0
    seen = set(fixed_sot)
    for mt, dt, who, lvl, odds in sot_ladder_raw:
        if (mt, who, lvl) in seen:
            continue
        seen.add((mt, who, lvl))
        sot.append({"Match": mt, "Date": dt, "player_name": who, "threshold": lvl,
                    "odds_decimal": round(odds / sot_scale.get(lvl, default_scale), 2)})
    if sot_report:
        ladder_scales.append(("SoT", sot_report))

    if ladder_scales:
        print()
        print("  ladder margin rescaled to the fixed level, POOLED across all fixtures "
              "(per-fixture range in brackets - it should barely move):")
        for what, report in ladder_scales:
            cells = "  ".join(f"{lvl}+ x{r[0]:.2f} [n={r[1]}, {r[2]:.2f}-{r[3]:.2f}]"
                              for lvl, r in sorted(report.items()))
            print(f"    {what:<9} {cells}")

    out = {}
    for key, rows in player_rows.items():
        if rows:
            # A player can arrive from both a dedicated market and a ladder (Anytime
            # Goalscorer and "Goals 1+" price the same thing); keep one row per player.
            out[key] = (pd.DataFrame(rows)
                        .drop_duplicates(subset=["player_name", "match_id"], keep="first")
                        .reset_index(drop=True))
    if clean_sheet:
        out["clean_sheet"] = pd.DataFrame(clean_sheet)
    if team_goals:
        out["team_goals"] = pd.DataFrame(team_goals)
    if clean_sheet_f2:
        out["clean_sheet_f2"] = pd.DataFrame(clean_sheet_f2)
    if team_goals_f2:
        out["team_goals_f2"] = pd.DataFrame(team_goals_f2)
    if saves:
        out["gk_saves_raw"] = pd.DataFrame(saves)
    if sot:
        out["shots_on_target"] = pd.DataFrame(sot)
    return out, sorted(seen_markets)


def fill_gaps(frames, verbose=True):
    """Derive markets Betway has not priced yet FROM the ones it has.

    Better than regenerating everything synthetically (tools/build_preseason_data.py),
    because each derivation is anchored to real odds for the SAME fixtures:

      2+ goals  exact, not a guess: P(2+) follows from P(1+) under the Poisson score
                curve the pipeline already uses (config-blessed, see model.poisson_score2)
      assists   calibrated: where Betway prices BOTH assists and goalscorer, measure the
                real assist:score ratio and apply it to fixtures missing assists. Falls
                back to the 1.132 convention ratio measured from real markets in 2026-08
                if no fixture has both.
      saves     no anchor exists — leave the placeholder alone rather than invent one.

    Anything derived here is marked so it never masquerades as scraped data.
    """
    from fpl_pipeline import model
    from fpl_pipeline.markets import implied
    model.load_coefficients()
    margin = config.MARGIN_PLAYER if hasattr(config, "MARGIN_PLAYER") else 1.08

    derived = []

    # --- 2+ goals from anytime goalscorer, via the Poisson curve ---
    # PER PLAYER, not all-or-nothing. Betway's ladder prices 2+ for only a subset (~70 of
    # 417), so an "is score2 present?" check leaves everyone else with a goal price and no
    # 2+ price at all — worse than the old fully-derived column.
    if "score1" in frames:
        s1 = frames["score1"]
        have = frames.get("score2", pd.DataFrame(columns=s1.columns))
        priced = set(zip(have["player_name"], have["match_id"]))
        gap = s1[~pd.Series(list(zip(s1["player_name"], s1["match_id"])),
                            index=s1.index).isin(priced)]
        if not gap.empty:
            p2 = model.poisson_score2(implied(gap["odds_decimal"], margin).clip(1e-6, 0.95))
            frames["score2"] = pd.concat([have, pd.DataFrame({
                "player_name": gap["player_name"], "match_id": gap["match_id"],
                "odds_decimal": (margin / p2.clip(1e-6, 0.95)).round(2)})], ignore_index=True)
            derived.append(f"score2 <- Poisson curve for {len(gap)} players Betway did not "
                           f"price at 2+ ({len(have)} real prices kept)")

    # --- assists for fixtures Betway has not priced yet ---
    if "score1" in frames:
        have = frames.get("assist", pd.DataFrame(columns=["player_name", "match_id", "odds_decimal"]))
        # fixture-level here (unlike score2): Betway either prices a fixture's assists or
        # does not, so a partially-priced fixture is not the failure mode to guard against
        priced = set(have["match_id"])
        missing = sorted(set(frames["score1"]["match_id"]) - priced)
        if missing:
            both = frames["score1"].merge(have, on=["player_name", "match_id"],
                                          suffixes=("_s", "_a"))
            if len(both) >= 20:
                ratio = float((implied(both["odds_decimal_a"], margin) /
                               implied(both["odds_decimal_s"], margin).clip(1e-6)).median())
                how = f"ratio {ratio:.3f} calibrated on {len(both)} real pairs"
            else:
                ratio, how = 1.132, "1.132 convention ratio (too few real pairs to calibrate)"
            src = frames["score1"][frames["score1"]["match_id"].isin(missing)]
            p_assist = (implied(src["odds_decimal"], margin) * ratio).clip(1e-6, 0.9)
            frames["assist"] = pd.concat([have, pd.DataFrame({
                "player_name": src["player_name"], "match_id": src["match_id"],
                "odds_decimal": (margin / p_assist).round(2)})], ignore_index=True)
            derived.append(f"assist <- {len(src)} rows for {len(missing)} unpriced fixtures, {how}")

    if verbose:
        print("\nderived from the real odds above (never scraped, never presented as such):")
        for d in derived or ["  nothing needed — every market priced"]:
            print(f"  {d}")
        if "gk_saves_raw" not in frames:
            n = len(frames.get("shots_on_target", []))
            print(f"  saves: no saves market on Betway; {n} shots-on-target rows captured "
                  "-> derive_saves() below" if n else
                  "  saves: no market and no shots data — placeholder stands")
    return frames


CONVERSION = 0.30      # league-wide shots-on-target -> goals rate; the one assumption here
TOP_SHOOTERS = 10      # outfield players per side — see the note in derive_saves()


def derive_saves(frames, conversion=CONVERSION, verbose=True):
    """Goalkeeper saves from the shots-on-target ladders, de-margined against a market anchor.

    Betway prices no saves market, but it does price ~43 players per match for 1+..4+ shots
    on target. Those cannot be used raw: summing P(X>=k) gives Arsenal 23.2 expected shots
    on target against a league average near 5, because 135 independent yes/no selections
    carry an enormous compounded overround.

    So anchor them. Clean-sheet prices give expected goals conceded directly
    (P(CS) = exp(-lambda_conceded)), and total match shots on target should be
    total goals / conversion. The ratio of the naive ladder sum to that anchor IS the
    margin, and dividing it out leaves per-team shot volumes on a real scale.

    The ladders then contribute what the anchor alone cannot: how the shots SPLIT between
    the sides. Teams vary in shots needed per goal, so a team's share of the match's shots
    is not its share of the goals, and the market prices that difference.

        saves(keeper) = shots on target faced - goals conceded
                      = lambda_SoT(opponent) - lambda_conceded(opponent's opponent)
    """
    sot, cs = frames.get("shots_on_target"), frames.get("clean_sheet")
    if sot is None or cs is None or sot.empty or cs.empty:
        return frames

    roster = pd.read_csv(os.path.join(config.OUTPUTS_DIR, "01_fpl_players.csv"))
    team_of = dict(zip(roster["name"], roster["team"]))
    keepers = roster[roster.get("position", "").astype(str).str.upper().isin(["GK", "GKP"])]
    gk_of = dict(zip(keepers["team"], keepers["name"]))

    # naive E[shots on target] per player = sum of P(X>=k) over the priced thresholds
    sot = sot.assign(p=1.0 / sot["odds_decimal"])
    per_player = sot.groupby(["Match", "player_name"], as_index=False)["p"].sum()
    per_player["team"] = per_player["player_name"].map(team_of)

    rows, notes = [], []
    for match, group in per_player.groupby("Match"):
        conceded = {r.team_name: -math.log(max(1.0 / r.clean_sheet_yes, 1e-6))
                    for r in cs[cs["match_name"] == match].itertuples()
                    if pd.notna(r.clean_sheet_yes)}
        # EQUAL PLAYERS PER SIDE. Betway does not price both squads evenly — one fixture
        # listed 24 players for one team and 13 for the other — and summing everyone lets
        # that choice move the split. Truncating Forest v Leeds to equal counts shifted the
        # share 63.9% -> 57.0%, nearly 7 points, purely from who happened to be listed.
        # Take each side's TOP_SHOOTERS most likely shooters: a team fields 10 outfielders,
        # keepers do not shoot, and anyone beyond that is a fringe name whose small
        # probability should not tip a keeper's projection.
        ranked = group.dropna(subset=["team"]).sort_values("p", ascending=False)
        naive = (ranked.groupby("team").head(TOP_SHOOTERS)
                       .groupby("team")["p"].sum())
        naive = naive[naive.index.isin(conceded)]
        if len(naive) != 2 or len(conceded) != 2:
            continue

        anchor = sum(conceded.values()) / conversion          # true total shots on target
        margin = naive.sum() / anchor                          # the overround, measured
        lam_sot = naive / margin                               # per team, real scale

        date = cs[cs["match_name"] == match]["date"].iloc[0]
        for team in conceded:                                  # this team's keeper
            opponent = next(t for t in conceded if t != team)
            faced = lam_sot.get(opponent)
            # goals the opponent scores = goals THIS team concedes
            lam_saves = max(faced - conceded[team], 0.1)
            rows.append({"Match": match, "Date": date, "Team": team,
                         "Goalkeeper": gk_of.get(team, ""),
                         "3+ Saves": round(1 / max(poisson_tail(lam_saves, 3), 1e-4), 2),
                         "6+ Saves": round(1 / max(poisson_tail(lam_saves, 6), 1e-4), 2)})
        notes.append(f"{match}: margin {margin:.2f}x, "
                     + ", ".join(f"{t} {lam_sot[t]:.1f} SoT" for t in lam_sot.index))

    if rows:
        frames["gk_saves"] = pd.DataFrame(rows)
        if verbose:
            print(f"\nsaves derived for {len(rows)} keepers "
                  f"(conversion {conversion:.0%}, margin measured per match):")
            for n in notes[:4]:
                print(f"  {n}")
            if len(notes) > 4:
                print(f"  ... and {len(notes) - 4} more")
    return frames


def poisson_tail(lam, k):
    """P(X >= k) for Poisson(lam)."""
    cum = sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(k))
    return max(1.0 - cum, 1e-6)


FILES = {"assist2": "sportsbet_two_assists_odds.csv",
         "wdw": "sportsbet_win_draw_win_odds.csv",
         "score1": "sportsbet_goalscorer_odds.csv",
         "score2": "sportsbet_two_goals_odds.csv",
         "assist": "sportsbet_assist_odds.csv",
         "yellow": "sportsbet_booking_odds.csv",
         "clean_sheet": "sportsbet_clean_sheet_odds.csv",
         "team_goals": "sportsbet_team_goals_odds.csv",
         "gk_saves": "sportsbet_goalkeeper_saves_odds.csv"}
# Next-gameweek team markets. Left header-only when F2 is not priced yet, so the pipeline's
# model fallback projects F2 clean sheets / concede from the F2 win odds (see players.py).
F2_FILES = {"clean_sheet_f2": "sportsbet_clean_sheet_odds_f2.csv",
            "team_goals_f2": "sportsbet_team_goals_odds_f2.csv"}

# Betway prices whole matches at a time, so a partial scrape must NOT wipe the synthetic rows for
# the matches it didn't price. Upsert each market: real where Betway prices it, synthetic
# (build_synthetic_gw) everywhere else — but BETWAY-AUTHORITATIVE PER MATCH. Player-props are kept
# only for players whose TEAM Betway didn't price at all; once Betway prices a player's team, its
# listed players are the whole truth for that team and any teammate it omitted goes to NA (Betway
# didn't rate him a scorer). Team markets upsert directly on their team column (already per-match).
PLAYER_PROP = {"score1": "match_id", "score2": "match_id", "assist": "match_id",
               "assist2": "match_id", "yellow": "match_name"}   # market -> match column
TEAM_KEY = {"clean_sheet": "team_name", "team_goals": "Team", "gk_saves": "Team"}


def _player_team_map():
    """name -> team from the roster snapshot, to resolve which teams Betway priced."""
    r = pd.read_csv(os.path.join(config.OUTPUTS_DIR, "01_fpl_players.csv"))
    return r.drop_duplicates(subset="name").set_index("name")["team"]


def _priced_teams(priced_df, keycol, tmap):
    """Teams Betway priced = the two MAJORITY teams of each priced match (grouped by the match
    column), NOT a per-player roster lookup — so a loan player registered to one club but playing
    for another (Disasi @ Chelsea, on loan at Palace) can't flip his parent club to 'priced'."""
    if keycol not in priced_df.columns:
        return set(priced_df["player_name"].map(tmap).dropna())
    team = priced_df["player_name"].map(tmap)
    out = set()
    for _, idx in priced_df.groupby(keycol).groups.items():
        out.update(team.loc[idx].value_counts().head(2).index)
    return out


def write(frames, dry_run=False):
    print("\nmarkets found and written (everything else keeps its current file):")
    tmap = _player_team_map() if any(k in PLAYER_PROP for k in frames) else None
    for key, fname in FILES.items():
        path = os.path.join(config.SPORTSBET_DIR, fname)
        if key in frames:
            out = frames[key]
            # Keep synthetic rows for matches Betway didn't price; Betway-authoritative per match.
            if os.path.exists(path):
                old = pd.read_csv(path)
                if key in PLAYER_PROP and "player_name" in old.columns and tmap is not None:
                    priced_teams = _priced_teams(out, PLAYER_PROP[key], tmap)   # majority teams of priced matches
                    kept = old[~old["player_name"].map(tmap).isin(priced_teams)]  # unknown team -> kept (safe)
                    out = pd.concat([out, kept], ignore_index=True)
                elif key in TEAM_KEY and TEAM_KEY[key] in old.columns and TEAM_KEY[key] in out.columns:
                    kcol = TEAM_KEY[key]
                    out = pd.concat([out, old[~old[kcol].isin(set(out[kcol]))]], ignore_index=True)
            if not dry_run:
                out.to_csv(path, index=False)
            kept_n = len(out) - len(frames[key])
            note = f"  ({len(frames[key])} priced, {kept_n} synthetic kept)" if kept_n > 0 else ""
            print(f"  WRITE  {fname:<40} {len(out):>5} rows{note}")
        else:
            exists = os.path.exists(path)
            print(f"  keep   {fname:<40} {'placeholder/previous kept' if exists else 'ABSENT'}")
    for key, fname in F2_FILES.items():
        path = os.path.join(config.SPORTSBET_DIR, fname)
        if key in frames:
            if not dry_run:
                frames[key].to_csv(path, index=False)
            print(f"  WRITE  {fname:<40} {len(frames[key]):>5} rows  (next gameweek)")
        else:
            print(f"  keep   {fname:<40} header-only — F2 model-projected")
    for key, fname, note in (
            ("gk_saves_raw", "betway_saves_raw.csv",
             "saves market appeared — raw dump for mapping"),
            ("shots_on_target", "betway_shots_on_target.csv",
             "shots-on-target ladders, for the saves estimate")):
        if key in frames and not dry_run:
            p = os.path.join(config.OUTPUTS_DIR, fname)
            frames[key].to_csv(p, index=False)
            print(f"  dump   {fname:<40} {len(frames[key]):>5} rows  ({note})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--har", help="parse a saved HAR instead of fetching")
    ap.add_argument("--limit", type=int, help="only this many fixtures")
    ap.add_argument("--skip-fixture", action="append", default=[], metavar="TERM",
                    help="drop fixtures whose 'Home vs. Away' label contains TERM (case-insensitive) "
                         "BEFORE the F1/F2 split — use to ignore a stray earlier-gameweek match "
                         "Betway still lists (repeatable). The kickoff-order split then lands the "
                         "first 10 remaining on F1 and the next 10 on F2.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-fill", action="store_true",
                    help="write only what Betway priced; do not derive the gaps")
    ap.add_argument("--conversion", type=float, default=CONVERSION,
                    help="shots-on-target to goals rate used to anchor the saves model")
    ap.add_argument("--delay", type=float, default=3.0,
                    help="seconds between fixtures (default 3; this runs weekly, so be polite)")
    args = ap.parse_args()

    if args.har:
        har = json.load(open(args.har, encoding="utf-8"))
        payloads = [json.loads(e["response"]["content"]["text"])
                    for e in har["log"]["entries"]
                    if "MarketGroupNamesAndMarketsForEvent" in e["request"]["url"]
                    and (e["response"]["content"].get("text") or "").startswith("{")]
        best = max(payloads, key=lambda p: len(p.get("outcomes", [])))
        rows = selections(best)
        print(f"{len(rows)} selections, {len({m for m, *_ in rows})} markets from {args.har}")
        df = pd.DataFrame(rows, columns=["market", "selection", "line", "odds_decimal"])
        out = os.path.join(config.OUTPUTS_DIR, "betway_odds.csv")
        df.to_csv(out, index=False)
        print(f"-> {os.path.relpath(out, config.ROOT)}")
        for m in sorted({m for m, *_ in rows}):
            if any(w in m.lower() for w in ("assist", "goalscorer", "clean sheet", "total goals", "book")):
                n = sum(1 for x in rows if x[0] == m)
                print(f"  {m:<44}{n:>4} selections")
        sys.exit()

    fx_all, wdw = fixtures(args.limit)
    print(f"{len(fx_all)} fixtures discovered, {len(wdw)} with 1X2 prices")
    if args.skip_fixture:
        terms = [t.lower() for t in args.skip_fixture]
        dropped = [r[1] for r in fx_all if any(t in r[1].lower() for t in terms)]
        fx_all = [r for r in fx_all if not any(t in r[1].lower() for t in terms)]
        for d in dropped:
            print(f"  --skip-fixture: ignoring '{d}' (kept {len(fx_all)} fixtures for the F1/F2 split)")
        if not dropped:
            print(f"  WARNING: --skip-fixture {args.skip_fixture} matched no fixture label")
    gw_of, fx = assign_f1_f2(fx_all)
    crosscheck_split([r for r in fx if gw_of.get(r[1]) == "f1"],
                     [r for r in fx if gw_of.get(r[1]) == "f2"])
    frames, seen = collect(fx, gw_of, delay=args.delay)
    if not wdw.empty:
        # Order the win/draw/win frame as [F1 block][F2 block] and drop the helper 'match'
        # column, so the pipeline's positional split (iloc[:10]=F1, iloc[10:20]=F2) lands
        # on the right gameweeks. Fixtures past F2 are dropped — never read from odds.
        order = {m: i for i, (_, m, _) in enumerate(fx)}
        wdw = wdw[wdw["match"].isin(order)].copy()
        wdw["__o"] = wdw["match"].map(order)
        wdw = (wdw.sort_values("__o").drop(columns=["match", "__o"]).reset_index(drop=True))
        frames["wdw"] = wdw
    if not args.no_fill:
        frames = fill_gaps(frames)
        frames = derive_saves(frames, conversion=args.conversion)
    write(frames, args.dry_run)
