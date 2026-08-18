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
            by_event[o["eventId"]][(o.get("displayName") or "").strip().lower()] = odds

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

        # per-team clean sheets: "<Team> To Keep A Clean Sheet" (both gameweeks)
        cs_bucket = clean_sheet if gw == "f1" else clean_sheet_f2
        for team in (home, away):
            m = next((k for k in by_market if "clean sheet" in k.lower()
                      and team.split()[0].lower() in k.lower() and "1st half" not in k.lower()), None)
            if m:
                p = {s.lower(): o for s, _, o in by_market[m]}
                cs_bucket.append({"match_name": match, "date": date, "team_name": team,
                                  "clean_sheet_yes": p.get("yes"), "clean_sheet_no": p.get("no")})

        # total goals -> the team-goals shape the pipeline reads (both gameweeks). Betway's
        # Total Goals is the MATCH total; concede_market() reads the file POSITIONALLY
        # (cols 4-5 = the 1.5/2+ line, cols 6-7 = the 3.5/4+ line) and keys on the Opponent
        # column — so emit both perspectives, and keep the 3.5 line in cols 6-7, or the home
        # side of every fixture gets no concede price and 4+ silently duplicates 2+.
        tg = defaultdict(dict)
        for sel, sbv, odds in by_market.get("Total Goals", []):
            if sbv:
                tg[sbv][sel.lower()] = odds
        if tg:
            tg_bucket = team_goals if gw == "f1" else team_goals_f2
            for a, b in ((home, away), (away, home)):
                row = {"Match": match, "Date": date, "Team": a, "Opponent": b}
                for want in ("1.5", "3.5"):
                    o = tg.get(want, {})
                    row[f"Team_Over_{want}"] = o.get("over")
                    row[f"Team_Under_{want}"] = o.get("under")
                for want in ("1.5", "3.5"):
                    o = tg.get(want, {})
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
        # LADDERS CARRY MORE MARGIN THAN THE FIXED MARKETS — measured 2026-08-19, summed
        # implied probability runs ~3.7x true for ladders against ~3.0x for fixed, on both
        # goals and assists. Left unadjusted, a player gap-filled from a ladder gets a
        # factor ~28% higher than an identical player priced from the fixed market. Uniform
        # inflation cancels in a factor, but this is a RELATIVE difference between players,
        # so it does not. Rescale each ladder to the fixed market's level using the players
        # priced in both, per fixture — the same self-calibrating trick as derive_saves().
        for market, levels in LADDERS.items():
            fixed_for = {lvl: k for lvl, k in levels.items()}
            scale = {}
            for lvl, key in fixed_for.items():
                fixed_px = {p["player_name"]: p["odds_decimal"]
                            for p in player_rows.get(key, []) if p["match_id"] == match}
                pairs = []
                for sel, _, odds in by_market.get(market, []):
                    nm, _, tail = sel.rpartition(" ")
                    if tail == f"{lvl}+":
                        who = player_name(nm)
                        if who in fixed_px:
                            pairs.append((1 / fixed_px[who], 1 / odds))
                if len(pairs) >= 8:
                    scale[lvl] = sum(f for f, _ in pairs) / sum(l for _, l in pairs)
            for sel, _, odds in by_market.get(market, []):
                name, _, tail = sel.rpartition(" ")
                if not tail.endswith("+") or not tail[:-1].isdigit():
                    continue
                lvl = int(tail[:-1])
                key = levels.get(lvl)
                if key:
                    adj = odds / scale.get(lvl, 1.0)      # scale<1 lengthens the price
                    player_rows[key].append({"player_name": player_name(name),
                                             "match_id": match,
                                             "odds_decimal": round(adj, 2)})
            if scale:
                ladder_scales.append((match, market.split()[1], scale))

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
        # The shots ladder is over-margined exactly like the goals/assists ladders —
        # measured 0.74-0.85 (mean ~0.78) across three fixtures and every threshold that
        # appears in both shapes. Rescale it to the fixed markets' level per fixture, or a
        # player gap-filled from the ladder carries ~28% more implied shots than a
        # neighbour priced from the fixed market, distorting the team split the saves model
        # depends on. The 5+ level exists ONLY in the ladder, so it inherits the mean of the
        # levels that could be measured.
        lad = defaultdict(dict)
        for sel, _, odds in by_market.get(SOT_LADDER, []):
            nm2, _, tail = sel.rpartition(" ")
            if tail.endswith("+") and tail[:-1].isdigit():
                lad[int(tail[:-1])][player_name(nm2)] = odds
        sot_scale = {}
        for lvl, priced in lad.items():
            fixed_px = {player_name(s): o
                        for s, _, o in by_market.get(f"Player {lvl}+ Shots On Target", [])}
            both = set(fixed_px) & set(priced)
            if len(both) >= 8:
                sot_scale[lvl] = (sum(1 / fixed_px[k] for k in both)
                                  / sum(1 / priced[k] for k in both))
        default_scale = (sum(sot_scale.values()) / len(sot_scale)) if sot_scale else 1.0
        for lvl, priced in sorted(lad.items()):
            k = sot_scale.get(lvl, default_scale)
            for who, odds in priced.items():
                if (who, lvl) in seen_sot:
                    continue
                seen_sot.add((who, lvl))
                sot.append({"Match": match, "Date": date, "player_name": who,
                            "threshold": lvl, "odds_decimal": round(odds / k, 2)})
        if sot_scale:
            ladder_scales.append((match, "SoT", {**sot_scale,
                                                 **{l: default_scale for l in lad
                                                    if l not in sot_scale}}))

        print(f"  [F1] {match:<32} {len(rows):>5} selections, {len(by_market):>3} markets")
        time.sleep(delay)

    if ladder_scales:
        print()
        print("  ladder margin rescaled to the fixed markets' level (per fixture, "
              "measured on players priced in both):")
        for m, what, sc in ladder_scales[:4]:
            print(f"    {m:<34} {what:<9} " + "  ".join(f"{k}+ x{v:.2f}" for k, v in sorted(sc.items())))
        if len(ladder_scales) > 4:
            print(f"    ... and {len(ladder_scales)-4} more")

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


def write(frames, dry_run=False):
    print("\nmarkets found and written (everything else keeps its current file):")
    for key, fname in FILES.items():
        path = os.path.join(config.SPORTSBET_DIR, fname)
        if key in frames:
            out = frames[key]
            # Saves are derived per fixture and only where shots-on-target markets exist,
            # so a plain write would delete the placeholder rows for every OTHER keeper.
            # Upsert on team instead: real where we have it, placeholder everywhere else.
            if key == "gk_saves" and os.path.exists(path):
                old = pd.read_csv(path)
                if "Team" in old.columns:
                    kept = old[~old["Team"].isin(set(out["Team"]))]
                    out = pd.concat([out, kept], ignore_index=True)
            if not dry_run:
                out.to_csv(path, index=False)
            note = (f"  ({len(frames[key])} derived, {len(out) - len(frames[key])} placeholder kept)"
                    if key == "gk_saves" and len(out) > len(frames[key]) else "")
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
    gw_of, fx = assign_f1_f2(fx_all)
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
