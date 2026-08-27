"""Odds provenance manifest: what is REAL vs SYNTHETIC right now, per market.

The sportsbet/*.csv files are a moving mix — build_synthetic_gw seeds them synthetic, betway.py
overrides priced fixtures, card ingests fill bookings — and nothing durably recorded which was which.
That cost us a wrong "it's all synthetic" call. This manifest is the single source of truth: each
writer stamps the markets it touches, keyed by the sportsbet FILENAME (the one id every tool shares).

  build_synthetic_gw -> reset_synthetic(gw, [...files...])   every seeded market = synthetic
  betway.py          -> mark(file, "real", "betway", ...)    each priced market = real
  bet365/ladbrokes   -> mark(file, "synthetic"|"real", ...)  bookings source (kept synthetic for now)

Read it with `state(file)` (real / synthetic / unknown) or `status()` (the whole dict). The --gw
archiver refuses to snapshot any market whose state is not "real" (see is_real)."""
import datetime
import json
import os

from . import config

# sportsbet filename -> short label, for human-readable status output
FRIENDLY = {
    "sportsbet_win_draw_win_odds.csv": "win/draw/win",
    "sportsbet_goalscorer_odds.csv": "goalscorer",
    "sportsbet_two_goals_odds.csv": "two goals",
    "sportsbet_assist_odds.csv": "assist",
    "sportsbet_two_assists_odds.csv": "two assists",
    "sportsbet_booking_odds.csv": "bookings (yellow)",
    "sportsbet_clean_sheet_odds.csv": "clean sheet",
    "sportsbet_team_goals_odds.csv": "team goals",
    "sportsbet_goalkeeper_saves_odds.csv": "gk saves",
}


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _load():
    path = config.PROVENANCE_JSON               # read dynamically so tests/config overrides take
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {"gw": None, "updated": None, "markets": {}}


def _save(doc):
    doc["updated"] = _now()
    with open(config.PROVENANCE_JSON, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    return doc


def reset_synthetic(gw, files, source="build_synthetic_gw"):
    """Start a fresh manifest for a new synthetic build: every seeded market -> synthetic."""
    ts = _now()
    doc = {"gw": gw, "updated": None,
           "markets": {f: {"state": "synthetic", "source": source, "detail": f"seeded GW{gw}", "ts": ts}
                       for f in files}}
    return _save(doc)


def mark(file, state, source, detail=""):
    """Stamp one market (by sportsbet filename). state is 'real' or 'synthetic'."""
    doc = _load()
    doc["markets"][file] = {"state": state, "source": source, "detail": detail, "ts": _now()}
    return _save(doc)


def state(file):
    return _load().get("markets", {}).get(file, {}).get("state", "unknown")


def is_real(file):
    return state(file) == "real"


def status():
    return _load()
