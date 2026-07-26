"""Single source of truth for player and team name normalisation.

All renames live in inputs/name_mappings.csv (columns: type, name, name_cleaned) —
consolidated from the old fpl_data/player_name_changes.csv, the FFS dict that lived in
starting_lineups.py, and the team fixes scattered across sportsbet.py / Overall Odds.
"""
import os

import pandas as pd

from . import config

NAME_MAPPINGS_CSV = os.path.join(config.INPUTS_DIR, "name_mappings.csv")

_cache = {}


def _mappings(path=None):
    path = path or NAME_MAPPINGS_CSV
    if path not in _cache:
        df = pd.read_csv(path)
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


def clear_cache():
    _cache.clear()
