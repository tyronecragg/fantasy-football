"""Fixture risk: the chance a current-gameweek match is actually PLAYED (weather, safety, etc.).

inputs/fixture_risk.csv rows: Gameweek, Team, play_prob, [replay_gw], note. Only rows for the current F1 gameweek
apply (a stale file has no effect). Each listed team's players have their F1 points multiplied by
play_prob - a postponed match scores 0, so the expected points shrink by the same factor.

Deliberately NOT applied to starting_lineups.csv: that file is P(named in the XI | the match is played),
is renormalised to 11 per team at consumption (which would undo a blanket multiplier), and is what the
source ledger scores. The archive also keeps the unscaled master (run.py), like team views: it records
the market as priced, not our postponement guess. F2-F8 are untouched (a postponed match moves to a
later date): unless replay_gw is given, in which case the postponed share, (1 - play_prob) x the raw
F1 points, is ADDED to that gameweek's XP column (the double gameweek the rearranged match creates).
"""
import os

import numpy as np
import pandas as pd

from . import config, names

SCALED_COLUMNS = ("F1 XP", "F1 XP Pre")


def load(gameweek, path=None):
    """{team: (play_prob, replay_gw or None)} for `gameweek` only. Missing/empty file or no rows -> {}."""
    path = path or config.FIXTURE_RISK_CSV
    if gameweek is None or not os.path.exists(path):
        return {}
    r = pd.read_csv(path, comment="#")
    if r.empty:
        return {}
    r = r[pd.to_numeric(r["Gameweek"], errors="coerce") == gameweek]
    replay = pd.to_numeric(r["replay_gw"], errors="coerce") if "replay_gw" in r.columns else pd.Series(np.nan, index=r.index)
    out = {}
    for team, p, rg in zip(names.apply_team_names(r["Team"].astype(str).str.strip()),
                           pd.to_numeric(r["play_prob"], errors="coerce"), replay):
        if np.isfinite(p):
            out[team] = (float(np.clip(p, 0.0, 1.0)), int(rg) if np.isfinite(rg) else None)
    return out


def apply(master, risk, gameweek=None):
    """Copy of `master` with each listed team's F1 points scaled by its play_prob; if a replay_gw is
    given (and `gameweek` is known), (1 - play_prob) x the raw F1 XP is added to that week's XP column.
    Returns (master, applied) with applied = {team: (play_prob, replay column or None)}."""
    if not risk:
        return master, {}
    m = master.copy()
    if "F1 XP" in m.columns:
        # Unadjusted GW points for picking the XI / bench order / captain: Tyrone starts postponement-risk
        # players as normal and relies on autosub; the 0.95 only steers transfers and squad choice.
        m["F1 XP Raw"] = pd.to_numeric(m["F1 XP"], errors="coerce")
    applied = {}
    for team, val in risk.items():
        p, replay_gw = val if isinstance(val, tuple) else (val, None)
        rows = m["Team"] == team
        if not rows.any():
            continue
        raw_f1 = pd.to_numeric(m.loc[rows, "F1 XP"], errors="coerce") if "F1 XP" in m.columns else None
        for c in SCALED_COLUMNS:
            if c in m.columns:
                m.loc[rows, c] = pd.to_numeric(m.loc[rows, c], errors="coerce") * p
        col = None
        if replay_gw is not None and gameweek is not None and raw_f1 is not None:
            k = replay_gw - gameweek + 1
            if k >= 2 and f"F{k} XP" in m.columns:
                col = f"F{k} XP"
                m.loc[rows, col] = pd.to_numeric(m.loc[rows, col], errors="coerce") + (1 - p) * raw_f1
        applied[team] = (p, col)
    return m, applied
