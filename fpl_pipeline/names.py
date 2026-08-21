"""Single source of truth for player and team name normalisation.

All renames live in inputs/name_mappings.csv (columns: type, name, name_cleaned) —
consolidated from the old fpl_data/player_name_changes.csv, the FFS dict that lived in
starting_lineups.py, and the team fixes scattered across sportsbet.py / Overall Odds.
"""
import os
import unicodedata

import pandas as pd

from . import config

NAME_MAPPINGS_CSV = os.path.join(config.INPUTS_DIR, "name_mappings.csv")
# Bet365 markets are collected by hand and spell some players differently from Betway.
# They get their OWN mapping so fixing a Bet365 name can never disturb how the roster is
# cleaned for the Betway markets (name_mappings.csv above).
BET365_NAME_MAPPINGS_CSV = os.path.join(config.INPUTS_DIR, "bet365_name_mappings.csv")

_cache = {}


def _mappings(path=None):
    from .io_utils import read_csv_tolerant

    path = path or NAME_MAPPINGS_CSV
    if path not in _cache:
        df = read_csv_tolerant(path)
        _cache[path] = {
            kind: dict(zip(sub["name"], sub["name_cleaned"]))
            for kind, sub in df.groupby("type")
        }
    return _cache[path]


def player_map(path=None):
    return _mappings(path).get("player", {})


def team_map(path=None):
    return _mappings(path).get("team", {})


def apply_player_names(series, path=None):
    """Map raw player names to canonical FPL names; unknown names pass through."""
    m = player_map(path)
    return series.map(m).fillna(series)


def apply_team_names(series, path=None):
    m = team_map(path)
    return series.map(m).fillna(series)


def apply_bet365_names(series, path=None):
    """Map raw Bet365 names to canonical FPL names via the Bet365-only mapping; unknown names
    pass through. Separate from apply_player_names so it never touches the Betway markets."""
    return apply_player_names(series, path or BET365_NAME_MAPPINGS_CSV)


def _norm(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().casefold().strip()


def resolve_to_roster(series, roster_names):
    """Rewrite names to their exact roster spelling via an accent/case-insensitive match.

    Exact roster names pass straight through; a name whose normalised form matches exactly one
    roster player is rewritten to that player's canonical spelling (so a stray accent — 'Aurele'
    vs 'Aurele' with the diacritic — can't silently drop a card); anything ambiguous (two
    players share a normalised form) or unknown is left untouched for the reconciler to flag."""
    roster_set = set(roster_names)
    norm_map = {}
    for n in roster_names:
        norm_map.setdefault(_norm(n), []).append(n)

    def fix(n):
        if n in roster_set:
            return n
        hits = norm_map.get(_norm(n))
        return hits[0] if hits and len(hits) == 1 else n

    return series.map(fix)


def clear_cache():
    _cache.clear()
