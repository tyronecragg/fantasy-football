# -*- coding: utf-8 -*-
"""Predicted-XI history + source-accuracy scorer — the 'earning trust' ledger.

Each source's predicted starting XI is frozen per gameweek in inputs/source_history/predicted_xis.csv
(long format: Season, Gameweek, Source, Team, Player). Once the gameweek plays, the actual XIs
(By Gameweek/GW*/lineups.csv, is_starting) let us score every source head-to-head — so trust is
EARNED by measurement, not assumed (see inputs/curation_sources.md "Earning trust").

    env/Scripts/python tools/source_history.py --seed-ffs --gw N          # freeze FFS staging as a source
    env/Scripts/python tools/source_history.py --seed-ours --gw N         # freeze our curated XI (top-11 F1)
    env/Scripts/python tools/source_history.py --capture-csv F.csv --gw N # freeze external sources from a CSV
    env/Scripts/python tools/source_history.py --score --gw N             # score frozen sources vs actual

External sources (RotoWire, All About FPL, the deadline sweep) are frozen with --capture-csv from a
Source,Team,Player file (one row per predicted starter) — the turnkey weekly path. Names are resolved
to the FPL roster; anything that doesn't match is printed so a typo can't silently rot the ledger.
Re-running a (source, gameweek) replaces its rows (upsert). GW1's back-fill (before this CLI existed)
is tools/seed_gw1_sources.py, kept as a worked example.
"""
import argparse
import os
import sys
import unicodedata

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fpl_pipeline import config, ingest, names  # noqa: E402

HIST = os.path.join(ROOT, "inputs", "source_history", "predicted_xis.csv")
COLS = ["Season", "Gameweek", "Source", "Team", "Player"]


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).casefold().strip()


def _squash(s):
    """Fold accents/case and drop everything but letters/digits — 'Lewis-Skelly' -> 'lewisskelly'."""
    return "".join(c for c in _norm(s) if c.isalnum())


# Source team labels -> roster/FPL-canonical, so every source aligns to the actual-XI vocabulary
# (else e.g. FFS's "Brighton and Hove Albion" never matches the data's "Brighton" and goes unscored).
TEAM_CANON = {
    "brighton and hove albion": "Brighton", "brighton & hove albion": "Brighton",
    "manchester city": "Man City", "manchester united": "Man Utd", "manchester utd": "Man Utd",
    "newcastle united": "Newcastle", "nottingham forest": "Nott'm Forest",
    "tottenham hotspur": "Spurs", "tottenham": "Spurs", "leeds united": "Leeds",
    "afc bournemouth": "Bournemouth", "wolverhampton wanderers": "Wolves", "west ham united": "West Ham",
}


# AAF/RotoWire name a few players so tersely (or ambiguously) that no team-scoped rule is safe.
# Keyed by (roster team name, source spelling) -> roster canonical. Extend as new sources land.
ALIASES = {
    ("Arsenal", "Gabriel"): "Gabriel Magalhaes",   # not Martinelli / Jesus
    ("Leeds", "DCL"): "Dominic Calvert-Lewin",
    ("Man City", "Nunes"): "Matheus Nunes",         # not Vitor Nunes
    ("Man Utd", "Tielemens"): "Youri Tielemans",    # source misspelling
    ("Sunderland", "O'Nein"): "Luke O'Nien",        # source misspelling
    ("Hull City", "Hjerto-Dahl"): "Jens Hjertø-Dahl",
}


class Resolver:
    """Resolve a source's player label to the roster canonical, scoped by team.

    Full-name sources (FFS) match by folded full string; surname/nickname sources (AAF, RotoWire)
    match a label against the roster players ON THAT TEAM — unique token/surname/substring hit wins,
    ambiguous or missing is flagged so it can't silently rot the ledger.
    """

    def __init__(self):
        r = ingest.load_fpl_players()[["name", "team"]].copy()
        r["name_m"] = names.apply_player_names(r["name"])
        self.by_norm = {}
        for _, row in r.iterrows():
            self.by_norm.setdefault(_norm(row.name_m), row["name"])
            self.by_norm.setdefault(_norm(row["name"]), row["name"])
        self.by_team = {}
        self._teams = sorted(r["team"].unique())
        for _, row in r.iterrows():
            self.by_team.setdefault(_norm(row.team), []).append(
                (row["name"], _squash(row.name_m), {_squash(t) for t in str(row.name_m).split()}))

    def canon_team(self, team):
        """Map a source's team label to the roster/FPL-canonical name (else return it unchanged)."""
        if team in self._teams:
            return team
        return TEAM_CANON.get(_norm(team), team)

    def resolve(self, team, player, player_mapped=None):
        if (team, player) in ALIASES:
            return ALIASES[(team, player)]
        for cand in (player_mapped, player):     # try the mapper's spelling, then the raw one
            if cand is not None:
                full = self.by_norm.get(_norm(cand))
                if full is not None:
                    return full
        cands = self.by_team.get(_norm(team), [])
        for cand in (player_mapped, player):
            if cand is None:
                continue
            aq = _squash(cand)
            hits = [c for c, sq, toks in cands if aq == sq or aq in toks]        # exact token / full
            if not hits:
                hits = [c for c, sq, toks in cands if len(aq) >= 4 and aq in sq]  # substring fallback
            if len(set(hits)) == 1:
                return hits[0]
        return None


def capture(season, gw, source, pairs, resolver=None):
    """Freeze a source's XI. `pairs` = iterable of (team, player). Upserts (season, gw, source)."""
    resolver = resolver or Resolver()
    pairs = list(pairs)
    # Route names through the pipeline's own mapper first (accent/alias fixes), keeping the raw
    # spelling too so a source is scored on the player it named, not a spelling we canonicalised.
    mapped = list(names.apply_player_names(pd.Series([p for _, p in pairs])))
    rows, unmatched = [], []
    for (team, player), player_m in zip(pairs, mapped):
        team = resolver.canon_team(team)
        canon = resolver.resolve(team, player, player_m)
        if canon is None:
            unmatched.append(f"{source}/{team}: {player}")
            canon = player
        rows.append({"Season": season, "Gameweek": gw, "Source": source, "Team": team, "Player": canon})
    os.makedirs(os.path.dirname(HIST), exist_ok=True)
    if os.path.exists(HIST):
        h = pd.read_csv(HIST)
        h = h[~((h.Season.astype(str) == str(season)) & (h.Gameweek == gw) & (h.Source == source))]
        h = pd.concat([h, pd.DataFrame(rows, columns=COLS)], ignore_index=True)
    else:
        h = pd.DataFrame(rows, columns=COLS)
    h.sort_values(["Season", "Gameweek", "Source", "Team"]).to_csv(HIST, index=False)
    print(f"  captured {source} GW{gw}: {len(rows)} players, {len({t for t, _ in pairs})} teams"
          + (f"  ({len(unmatched)} UNMATCHED)" if unmatched else ""))
    for u in unmatched:
        print(f"    UNMATCHED (fix the name): {u}")
    return unmatched


def seed_ffs(season, gw):
    f = pd.read_csv(os.path.join(ROOT, "inputs", "ffs_predicted_lineups.csv"))
    capture(season, gw, "FFS", list(zip(f["Team"], f["Player"])))


def seed_ours(season, gw):
    """Our predicted XI = the 11 highest F1 start-probs per team (what we effectively predicted)."""
    sl = pd.read_csv(os.path.join(ROOT, "inputs", "starting_lineups.csv"))
    pairs = []
    for team, g in sl.groupby("Team"):
        for p in g.sort_values("F1", ascending=False)["Player"].head(11):
            pairs.append((team, p))
    capture(season, gw, "Curated", pairs)


def capture_csv(season, gw, path):
    """Freeze one or more external sources from a CSV of Source,Team,Player (one row per starter).

    The turnkey weekly path for RotoWire / All About FPL / the sweep consensus: the deadline sweep
    writes this file, then `--capture-csv path --gw N` upserts each source. Prints UNMATCHED lines
    to fix (add an ALIASES entry or a name_mappings.csv row) before the row is trusted.
    """
    df = pd.read_csv(path)
    need = {"Source", "Team", "Player"}
    if not need.issubset(df.columns):
        raise SystemExit(f"{path} must have columns {sorted(need)} (got {list(df.columns)})")
    resolver = Resolver()
    for source, g in df.groupby("Source"):
        capture(season, gw, source, list(zip(g["Team"], g["Player"])), resolver=resolver)


def score(season, gw):
    """Score every frozen source for (season, gw) against the actual XIs in the FPL data."""
    if not os.path.exists(HIST):
        raise SystemExit("no source_history yet — seed some sources first")
    h = pd.read_csv(HIST)
    h = h[(h.Season.astype(str) == str(season)) & (h.Gameweek == gw)]
    if h.empty:
        raise SystemExit(f"no captured sources for {season} GW{gw}")

    gw_dir = os.path.join(config.FPL_DATA_DIR, "By Gameweek", f"GW{gw}")
    lp = os.path.join(gw_dir, "lineups.csv")
    if not os.path.exists(lp):
        raise SystemExit(f"actual lineups not available yet ({lp}) — score after the gameweek plays")
    teams = ingest.load_fpl_players()  # (unused directly, but ensures data present)
    tmap = pd.read_csv(os.path.join(gw_dir, "teams.csv"))
    code2name = dict(zip(tmap.iloc[:, 0], tmap[[c for c in tmap.columns if "name" in c.lower()][0]]))
    matches = pd.read_csv(os.path.join(gw_dir, "matches.csv"), usecols=["match_id", "tournament"])
    pl = set(matches[matches.tournament == "prem"].match_id)
    lu = pd.read_csv(lp, usecols=["match_id", "team_code", "player_name", "is_starting"])
    lu = lu[lu.match_id.isin(pl) & lu.is_starting & lu.player_name.notna()]
    actual = {}
    for _, r in lu.iterrows():
        actual.setdefault(str(code2name.get(r.team_code, r.team_code)), set()).add(_norm(r.player_name))

    print(f"\nSOURCE ACCURACY — {season} GW{gw} (predicted XI vs who actually started)")
    print(f"{'Source':<12}{'teams':>7}{'predicted':>11}{'correct':>9}{'hit rate':>10}")
    print("-" * 49)
    rows = []
    for source, g in h.groupby("Source"):
        pred_teams = [t for t in g.Team.unique() if any(_norm(t) == _norm(a) for a in actual)]
        hit = tot = 0
        for team, tg in g.groupby("Team"):
            act = next((actual[a] for a in actual if _norm(a) == _norm(team)), None)
            if act is None:
                continue
            for p in tg.Player:
                tot += 1
                hit += _norm(p) in act
        rows.append((source, len(pred_teams), tot, hit, hit / tot if tot else float("nan")))
    for s, nt, tot, hit, hr in sorted(rows, key=lambda x: (-(x[4] if x[4] == x[4] else -1))):
        print(f"{s:<12}{nt:>7}{tot:>11}{hit:>9}{hr:>9.1%}")
    print("\n(hit rate = of the players a source predicted to start, the fraction who actually did)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, required=True)
    ap.add_argument("--season", default=config.SEASON)
    ap.add_argument("--seed-ffs", action="store_true")
    ap.add_argument("--seed-ours", action="store_true")
    ap.add_argument("--capture-csv", metavar="PATH", help="freeze external sources from a Source,Team,Player CSV")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.seed_ffs:
        seed_ffs(a.season, a.gw)
    if a.seed_ours:
        seed_ours(a.season, a.gw)
    if a.capture_csv:
        capture_csv(a.season, a.gw, a.capture_csv)
    if a.score:
        score(a.season, a.gw)
    if not (a.seed_ffs or a.seed_ours or a.capture_csv or a.score):
        ap.error("nothing to do — pass --seed-ffs / --seed-ours / --capture-csv / --score")


if __name__ == "__main__":
    main()
