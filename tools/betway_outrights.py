# -*- coding: utf-8 -*-
"""Scrape Betway's Premier League OUTRIGHT markets (title / top-6 / relegation) into the consensus
odds files the pipeline reads. Companion to tools/betway.py (match markets).

    env/Scripts/python tools/betway_outrights.py            # live fetch
    env/Scripts/python tools/betway_outrights.py --har F.har # parse a saved capture instead

Writes RAW Betway decimal odds to inputs/{title,relegation,top6}_odds.csv as `book_1`; season_probs
then de-margins with the flat MARGIN_SEASON (=1.08), which matches Betway's measured ~8% outright
margin — so raw-odds + flat-1.08 reproduces the exact de-margin (per-team relegation Yes/No, and the
winners=K markets summing to K). Markets used:
  - title       : "Premier League - Winner"        (one market, team odds)
  - top6        : "Premier League - Top 6"          (winners=6 market, team odds)
  - relegation  : per-team "To Be Relegated - <Team>" YES odds — more current than the main
                  "Premier League - Relegation" market (which carries stale outliers, e.g. a
                  Bournemouth 3.0 vs the 2-way's 10.0). FALLS BACK to the main market for the few
                  safe teams (Arsenal/Liverpool/…) that get no per-team market.
"""
import argparse
import json
import os
import sys

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fpl_pipeline import config, ingest  # noqa: E402
from fpl_pipeline.names import apply_team_names  # noqa: E402
from tools.betway import BASE, COMMON, HEADERS  # noqa: E402  (reuse the working client)

OUTRIGHT_PATH = "FeedsOutright/Events/OutrightLeagues"
OUTRIGHT_PARAMS = {"regionId": "england", "sportId": "soccer"}
PL_EVENT = "Premier League"
FILES = {"title": ("Premier League - Winner", "title_odds.csv"),
         "top6": ("Premier League - Top 6", "top6_odds.csv")}


def canon(name):
    return apply_team_names(pd.Series([str(name).strip()]))[0]


def fetch(har=None):
    if har:
        h = json.load(open(har, encoding="utf-8"))
        for e in h["log"]["entries"]:
            if "FeedsOutright" in e["request"]["url"] and e["response"]["content"].get("text"):
                return json.loads(e["response"]["content"]["text"])
        raise SystemExit(f"no FeedsOutright entry with a body in {har} (a 304 cache-hit has none — "
                         "recapture with a hard refresh)")
    r = requests.get(f"{BASE}/{OUTRIGHT_PATH}", params={**COMMON, **OUTRIGHT_PARAMS},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def parse(d):
    price = {p["outcomeId"]: p["priceDecimal"] for p in d["prices"]}
    eid = next((e["eventId"] for e in d["events"] if e["name"] == PL_EVENT), None)
    if eid is None:
        raise SystemExit("no Premier League event in the outrights feed")
    ocs = [o for o in d["outcomes"] if o["eventId"] == eid]
    mkts = {m["marketId"]: m["name"] for m in d["markets"] if m["eventId"] == eid}

    def market_outcomes(market_name):
        mid = next((k for k, v in mkts.items() if v == market_name), None)
        return {o["name"]: price.get(o["outcomeId"]) for o in ocs if o["marketId"] == mid} if mid else {}

    out = {}
    for key, (mname, _) in FILES.items():
        out[key] = {canon(n): p for n, p in market_outcomes(mname).items() if p}

    # relegation: per-team Yes odds (more current), main-market fallback for uncovered teams
    releg, fb = {}, 0
    for mid, mname in mkts.items():
        if "To Be Relegated -" in mname:
            oo = {o["name"]: price.get(o["outcomeId"]) for o in ocs if o["marketId"] == mid}
            if oo.get("Yes"):
                releg[canon(mname.split("To Be Relegated - ")[1])] = oo["Yes"]
    for n, p in market_outcomes(PL_EVENT + " - Relegation").items():
        if p and canon(n) not in releg:
            releg[canon(n)] = p
            fb += 1
    out["relegation"] = releg
    return out, fb


def write_book(odds, fname):
    df = (pd.DataFrame(sorted(odds.items()), columns=["Team", "book_1"])
          .drop_duplicates("Team", keep="first"))
    df.to_csv(os.path.join(config.INPUTS_DIR, fname), index=False)
    return df


def fill_missing_teams(out, teams):
    """A team Betway doesn't list in a market is at a SETTLED BOUNDARY — 0% or 100%. Which one flips
    with the season: early on "not listed" = no-hoper (0%); late on a settled market can mean CLINCHED
    (100%). Infer it from the team's STRENGTH in the OTHER markets (a clinched-top6 side is strong; an
    eliminated one is weak): rank all teams by strength, then a missing team is 100% if it sits in the
    market's achieved band (title=strongest, top6=top 6, relegation=weakest 3), else 0%. Fills `out` in
    place with sentinel odds (0%) or ~1/MARGIN_SEASON odds (100%); returns {market: [f"team X%", ...]}."""
    m = config.MARGIN_SEASON
    sentinel = {"title": config.SENTINEL_TITLE_TOP6, "top6": config.SENTINEL_TITLE_TOP6,
                "relegation": config.SENTINEL_RELEGATION}
    hundred_odds = round(1 / m, 3)                      # de-margins to ~1.0 (100%)
    band = {"title": 1, "top6": 6, "relegation": 3}

    def prob(market, t):
        o = out[market].get(t)
        return (1 / o / m) if o else None

    def strength(t):                                   # attacking quality: title + top6 (missing terms = 0)
        return (prob("title", t) or 0.0) + (prob("top6", t) or 0.0)   # strong tops the rank, weak bottoms it

    rank = {t: i for i, t in enumerate(sorted(teams, key=strength, reverse=True))}  # 0 = strongest
    report = {}
    for market in out:
        info = []
        for t in sorted(teams - set(out[market])):
            if market == "relegation":
                is_100 = rank[t] >= len(teams) - band["relegation"]     # among the weakest -> relegated
            else:
                is_100 = rank[t] < band[market]                         # among the strongest -> achieved
            out[market][t] = hundred_odds if is_100 else sentinel[market]
            info.append(f"{t} {'100%' if is_100 else '0%'}")
        report[market] = info
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--har", help="parse a saved HAR instead of a live fetch")
    ap.add_argument("--dry-run", action="store_true", help="print, don't write the CSVs")
    a = ap.parse_args()

    out, fb = parse(fetch(a.har))
    teams = set(ingest.load_fpl_players()["team"].dropna())
    for market, info in fill_missing_teams(out, teams).items():
        if info:
            print(f"  WARNING [{market}]: {len(info)} team(s) not listed, inferred from other markets "
                  f"-> {', '.join(info)}  (verify late-season boundary calls)")

    fnames = {"title": "title_odds.csv", "top6": "top6_odds.csv", "relegation": "relegation_odds.csv"}
    print(f"Betway PL outrights ({'HAR' if a.har else 'live'}):")
    for key, fname in fnames.items():
        df = write_book(out[key], fname) if not a.dry_run else pd.DataFrame(sorted(out[key].items()),
                                                                           columns=["Team", "book_1"])
        note = f" ({fb} via main-market fallback)" if key == "relegation" and fb else ""
        print(f"  {fname:<22} {len(df):>2} teams{note}"
              + ("  [dry-run]" if a.dry_run else ""))
        # de-margined probabilities the pipeline will derive (implied = 1/odds/MARGIN_SEASON)
        top3 = sorted(((t, 1 / o / config.MARGIN_SEASON) for t, o in out[key].items()),
                      key=lambda x: -x[1])[:3]
        print("     top 3: " + ", ".join(f"{t} {p * 100:.1f}%" for t, p in top3))
        if a.dry_run:
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()
