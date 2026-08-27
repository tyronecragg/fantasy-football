"""Ingest hand-collected Ladbrokes 'To Be Booked' odds into sportsbet_booking_odds.csv.

Ladbrokes lists nearly the whole squad per match, so for every COMPLETE match this writes the
captured players' real odds AND fills every other roster player in the two teams with that
match's MAX (longest) odds — the least-bookable captured price, a low, conservative booking
chance. Matches with too little data (a stray keeper or two) are left entirely synthetic.

    python tools/ladbrokes_cards.py                         # DEFAULT: fetch the market live
    python tools/ladbrokes_cards.py --dry-run
    python tools/ladbrokes_cards.py --har  odds_raw/lad.har # fallback: saved HAR / detail-service JSON
    python tools/ladbrokes_cards.py --input odds_raw/ladbrokes.html   # fallback: legacy saved HTML

The eurobet detail-service API answers 200 server-to-server (no Cloudflare clearance needed), so a
bare run fetches directly; --har/--input remain as fallbacks. All feed the same resolve + game-max
fill + upsert, and mark the bookings market REAL in the provenance manifest.

Names arrive as 'Surname Given...' (and occasionally already 'Given Surname'); each is resolved
by trying candidate word-orders against the roster, then the shared card name-map
(inputs/bet365_name_mappings.csv) and an accent/case match. The two teams in a match are inferred
from the resolved players themselves, so no team-name parsing is needed. Upserts by player: the
pipeline never reads match_name, so only the player and price matter.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config, names, provenance  # noqa: E402

BOOKING_FILE = "sportsbet_booking_odds.csv"
BOOKING_CSV = os.path.join(config.SPORTSBET_DIR, BOOKING_FILE)
YELLOW_BET_ID = 12064   # eurobet 'PLAYER TO GET A YELLOW CARD'
# The eurobet detail-service endpoint for the PL yellow-card market. Server-to-server it answers 200
# without Cloudflare clearance (unlike the browser HTML), so --fetch pulls it live; --har stays as a
# fallback if they ever gate it. The trailing id is the market's stable alias code.
YELLOW_URL = ("https://www.ladbrokes.be/detail-service/sport-schedule/services/meeting/"
              "calcio/ing-premier-league/tutte/ammonito-si-47-no_1592484600768?prematch=1&live=0")
FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:154.0) Gecko/20100101 Firefox/154.0",
    "Accept": "application/json, text/plain, */*",
    "X-EB-MarketId": "5", "X-EB-PlatformId": "1", "X-EB-Accept-Language": "en_BE",
}
ROSTER_CSV = os.path.join(config.OUTPUTS_DIR, "01_fpl_players.csv")
MIN_PLAYERS = 8   # fewer resolved than this -> treat the match as incomplete, leave it synthetic

MATCH_SPLIT = 'event-name prematch-name"><span>'
TEAMS_RE = re.compile(r"^([^<]+)</span><span>([^<]+)</span>")
# a player's title, then the FIRST quota that follows it — the 'Yes' price (the 'No' 1.00 comes after)
PLAYER_RE = re.compile(r'title="([^"]+)"[\s\S]*?<div class="quota"><div>([0-9]+(?:\.[0-9]+)?)</div>')


def candidates(raw):
    """Plausible 'Given Surname' orderings of a Ladbrokes 'Surname Given...' name."""
    toks = raw.split()
    if len(toks) == 1:
        return [raw]
    out, seen = [], set()
    for c in (raw,                                    # already Given Surname
              " ".join(toks[1:] + toks[:1]),          # surname is the first token
              toks[-1] + " " + " ".join(toks[:-1]),   # given name is the last token
              " ".join(reversed(toks))):              # full reverse
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def make_resolver(roster_names):
    norm_map = {}
    for n in roster_names:
        norm_map.setdefault(names._norm(n), []).append(n)

    def resolve(raw):
        for cand in candidates(raw):
            mapped = names.apply_bet365_names(pd.Series([cand])).iloc[0]
            hits = norm_map.get(names._norm(mapped))
            if hits and len(hits) == 1:
                return hits[0]
        return None
    return resolve


def parse_matches(html):
    out = []
    for seg in html.split(MATCH_SPLIT)[1:]:
        m = TEAMS_RE.match(seg)
        if not m:
            continue
        players = [(n.strip(), float(o)) for n, o in PLAYER_RE.findall(seg)]
        out.append((m.group(1).strip(), m.group(2).strip(), players))
    return out


def _yellow_json_blobs(obj):
    """Yield the eurobet detail-service JSON object(s) that carry the yellow-card market, from
    either a HAR (log.entries[].response.content.text) or a raw detail-service JSON."""
    if isinstance(obj, dict) and "log" in obj:
        for e in obj["log"].get("entries", []):
            text = e.get("response", {}).get("content", {}).get("text", "")
            if text and ("ammonito" in text or "YELLOW CARD" in text):
                try:
                    yield json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    continue
    else:
        yield obj


def _parse_detail(doc):
    """eurobet detail-service doc (HAR or raw JSON) -> [(home, away, [(name, decimal_odds)...])...].
    oddValue is decimal odds x100 (285 -> 2.85); only the 'Yes' side (resultCode 1) is taken."""
    out = []
    for j in _yellow_json_blobs(doc):
        for dg in j.get("result", {}).get("dataGroupList", []):
            for item in dg.get("itemList", []):
                ev = item.get("eventInfo", {})
                home = ev.get("teamHome", {}).get("description", "").strip()
                away = ev.get("teamAway", {}).get("description", "").strip()
                players = []
                for bg in item.get("betGroupList", []):
                    if bg.get("betId") != YELLOW_BET_ID:
                        continue
                    for og in bg.get("oddGroupList", []):
                        raw = og.get("oddGroupDescription", "").strip()   # 'Surname Given'
                        yes = next((o.get("oddValue") for o in og.get("oddList", [])
                                    if o.get("resultCode") == 1), None)
                        if raw and yes:
                            players.append((raw, yes / 100.0))
                if players:
                    out.append((home, away, players))
    return out


def parse_har(path):
    return _parse_detail(json.load(open(path, encoding="utf-8")))


def fetch_yellow(url=YELLOW_URL):
    import requests
    r = requests.get(url, headers=FETCH_HEADERS, timeout=30)
    r.raise_for_status()
    return _parse_detail(r.json())


def main():
    ap = argparse.ArgumentParser(description="Ingest Ladbrokes booking odds into the pipeline CSV.")
    src = ap.add_mutually_exclusive_group()   # none given -> fetch live (the default)
    src.add_argument("--input", help="saved Ladbrokes 'To Be Booked' HTML file (fallback)")
    src.add_argument("--har", help="saved HAR / eurobet detail-service JSON (fallback)")
    ap.add_argument("--url", default=YELLOW_URL, help="override the fetch endpoint")
    ap.add_argument("--dry-run", action="store_true", help="show what would change, write nothing")
    args = ap.parse_args()

    if args.input:
        matches = parse_matches(open(args.input, encoding="utf-8").read())
        if not matches:
            sys.exit("No matches parsed — check the saved HTML is the 'To Be Booked' page.")
    elif args.har:
        matches = parse_har(args.har)
        if not matches:
            sys.exit("No yellow-card market found in the HAR/JSON (expected betId 12064 / 'ammonito').")
    else:                                       # default: fetch live
        matches = fetch_yellow(args.url)
        if not matches:
            sys.exit("Fetched OK but no yellow-card market found (endpoint or betId 12064 changed?).")

    ros = pd.read_csv(ROSTER_CSV)
    resolve = make_resolver(list(ros["name"]))
    team_players = ros.groupby("team")["name"].apply(list).to_dict()

    rows, unmatched = {}, []            # rows: player -> odds (player-keyed upsert)
    for home, away, players in matches:
        resolved = []
        for raw, odds in players:
            hit = resolve(raw)
            (resolved.append((hit, odds)) if hit else unmatched.append(raw))
        label = f"{home} v {away}"
        if len(resolved) < MIN_PLAYERS:
            print(f"  SKIP  {label:40} incomplete ({len(resolved)} players) -> stays synthetic")
            continue
        two_teams = [t for t, _ in Counter(ros.set_index("name")["team"].get(n)
                                           for n, _ in resolved).most_common(2)]
        game_max = max(o for _, o in resolved)
        captured = {n for n, _ in resolved}
        for n, o in resolved:
            rows[n] = o
        filled = 0
        for t in two_teams:
            for p in team_players.get(t, []):
                if p not in captured:
                    rows[p] = game_max
                    filled += 1
        print(f"  OK    {label:40} {len(resolved):>2} real + {filled:>2} @ max {game_max:g} "
              f"({', '.join(two_teams)})")

    if unmatched:
        uniq = sorted(set(unmatched))
        print(f"\n  UNMATCHED ({len(uniq)}) - non-FPL depth, or a spelling needing a mapping:")
        for u in uniq:
            print(f"    {u}")
        print("  If any SHOULD be an FPL player, add a row to inputs/bet365_name_mappings.csv and re-run.")

    new = pd.DataFrame({"match_name": "Ladbrokes GW", "date": "",
                        "player_name": list(rows), "odds_decimal": list(rows.values())})
    if args.dry_run:
        print(f"\n[dry-run] would upsert {len(new)} players into {os.path.basename(BOOKING_CSV)}.")
        return
    existing = pd.read_csv(BOOKING_CSV) if os.path.exists(BOOKING_CSV) else \
        pd.DataFrame(columns=["match_name", "date", "player_name", "odds_decimal"])
    kept = existing[~existing["player_name"].isin(new["player_name"])]
    out = pd.concat([kept, new], ignore_index=True)
    out.to_csv(BOOKING_CSV, index=False)
    n_matches = sum(1 for m in matches if len({resolve(r) for r, _ in m[2]} - {None}) >= MIN_PLAYERS)
    provenance.mark(BOOKING_FILE, "real", "ladbrokes",
                    f"{len(new)} players over {n_matches} priced matches (game-max fill)")
    print(f"\nWrote {len(new)} players ({len(existing) - len(kept)} replaced); {len(out)} rows total. "
          f"Bookings marked REAL (provenance).")


if __name__ == "__main__":
    main()
