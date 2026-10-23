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
import re
import sys
import unicodedata

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fpl_pipeline import config, ingest, names  # noqa: E402

HIST = os.path.join(ROOT, "inputs", "source_history", "predicted_xis.csv")
COLS = ["Season", "Gameweek", "Source", "Team", "Player"]

# Sources the archive expects every gameweek — Curated is ours (--seed-ours); FFS, Fantasy Football
# Pundit and RotoWire are the standing externals captured at each deadline ("compile the teams").
# score() flags any that are missing so a source can't silently fall out of the ledger the way
# RotoWire did through GW2. FFP promoted to a primary source 2026-10-08 (name kept as in GW2-3 rows).
MAIN_SOURCES = ["Curated", "FFS", "FantasyFootballPundit", "RotoWire"]


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s).replace("ø", "o").replace("Ø", "O"))
    return "".join(c for c in s if not unicodedata.combining(c)).casefold().strip()


def _squash(s):
    """Fold accents/case and drop everything but letters/digits — 'Lewis-Skelly' -> 'lewisskelly'."""
    return "".join(c for c in _norm(s) if c.isalnum())


_INITIAL = re.compile(r"^[A-Za-z]\.?$")
_GLUED = re.compile(r"^[A-Za-z]\.(?=\S)")
_PAREN = re.compile(r"^(.+?)\s*\((.+)\)$")


def _strip_initials(label):
    """Drop leading forename initials from a source label — RotoWire prints 'E. Haaland',
    'V. van Dijk', 'M. Lewis-Skelly', so match on the surname portion ('Haaland', 'van Dijk')."""
    label = _GLUED.sub("", str(label).strip())                 # FFP 'J.TIMBER' -> 'TIMBER'
    m = _PAREN.match(label)                                     # FFS 'MURPHY (JACOB)' -> 'JACOB MURPHY'
    if m:
        label = f"{m.group(2)} {m.group(1)}"
    parts = [p for p in label.split() if not _INITIAL.match(p)]
    return " ".join(parts) if parts else label


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
    ("Newcastle", "Will Osula"): "William Osula",     # RotoWire short first name
    ("Man City", "Nunes"): "Matheus Nunes",         # not Vitor Nunes
    ("Man Utd", "Tielemens"): "Youri Tielemans",    # source misspelling
    ("Sunderland", "O'Nein"): "Luke O'Nien",        # source misspelling
    ("Hull City", "Hjerto-Dahl"): "Jens Hjertø-Dahl",
    ("Arsenal", "RAYA MARTIN"): "David Raya",            # FFS full surname
    ("Bournemouth", "ANTÓNIO SILVA"): "António João Pereira de Albuquerque Tavares da Silva",
    ("Brentford", "YARMOLYUK"): "Yehor Yarmoliuk",       # FFS/RotoWire transliteration
    ("Brentford", "Yehor Yarmolyuk"): "Yehor Yarmoliuk",
    ("Brighton", "CHEMA ANDRÉS"): "José María Andrés Baixauli",
    ("Newcastle", "MILEY"): "Lewis Miley",
    ("Spurs", "SAVINHO"): "Sávio",                       # FFS uses his nickname
    ("Liverpool", "A.BECKER"): "Alisson Becker",
    ("Arsenal", "KONSA NGOYO"): "Ezri Konsa",           # FFS full surname
    ("Brighton", "Chema"): "José María Andrés Baixauli",  # RotoWire nickname
}
ALIASES = {(t, _norm(n)): v for (t, n), v in ALIASES.items()}  # case/accent-insensitive lookup


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
        if (team, _norm(player)) in ALIASES:
            return ALIASES[(team, _norm(player))]
        # Try the mapper's spelling, then the raw one, then each with leading initials stripped
        # ('E. Haaland' -> 'Haaland') so RotoWire-style labels resolve too.
        raw = [c for c in (player_mapped, player) if c is not None]
        cand_list = raw + [s for s in (_strip_initials(c) for c in raw) if s not in raw]
        for cand in cand_list:
            full = self.by_norm.get(_norm(cand))
            if full is not None:
                return full
        cands = self.by_team.get(_norm(team), [])
        m = re.match(r"^([A-Za-z])\.\s*\S", str(player).strip())
        ini = _norm(m.group(1)) if m else None
        for cand in cand_list:
            aq = _squash(cand)
            if not aq:
                continue
            hits = [c for c, sq, toks in cands if aq == sq or aq in toks]        # exact token / full
            if not hits:
                hits = [c for c, sq, toks in cands if len(aq) >= 4 and aq in sq]  # substring fallback
            if len(set(hits)) > 1 and ini:                                  # 'L.MILEY': use the initial
                hits = [c for c in hits if _norm(c).startswith(ini)]
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


def _actual_xis(gw_dir):
    """{team_name: {normalised starter names}} for a played GW. Prefers lineups.csv (is_starting);
    falls back to player_gameweek_stats.csv `starts==1` keyed to the FPL roster — the layout the
    current FPL-Core-Insights dumps use (they ship no lineups.csv, but a `starts` column instead)."""
    lp = os.path.join(gw_dir, "lineups.csv")
    if os.path.exists(lp):
        tmap = pd.read_csv(os.path.join(gw_dir, "teams.csv"))
        code2name = dict(zip(tmap.iloc[:, 0], tmap[[c for c in tmap.columns if "name" in c.lower()][0]]))
        matches = pd.read_csv(os.path.join(gw_dir, "matches.csv"), usecols=["match_id", "tournament"])
        pl = set(matches[matches.tournament == "prem"].match_id)
        lu = pd.read_csv(lp, usecols=["match_id", "team_code", "player_name", "is_starting"])
        lu = lu[lu.match_id.isin(pl) & lu.is_starting & lu.player_name.notna()]
        actual = {}
        for _, r in lu.iterrows():
            actual.setdefault(str(code2name.get(r.team_code, r.team_code)), set()).add(_norm(r.player_name))
        return actual
    pgs_path = os.path.join(gw_dir, "player_gameweek_stats.csv")
    if not os.path.exists(pgs_path):
        return {}
    pgs = pd.read_csv(pgs_path)
    if "starts" not in pgs.columns:
        return {}
    started = set(pd.to_numeric(pgs.loc[pd.to_numeric(pgs["starts"], errors="coerce").fillna(0) == 1, "id"],
                                errors="coerce").dropna().astype(int))
    roster = ingest.load_fpl_players().dropna(subset=["player_id"]).copy()
    roster["player_id"] = roster["player_id"].astype(int)
    roster = roster.set_index("player_id")
    actual = {}
    for pid in started:
        if pid in roster.index:
            r = roster.loc[pid]
            actual.setdefault(str(r["team"]), set()).add(_norm(r["name"]))
    return actual


def score(season, gw):
    """Score every frozen source for (season, gw) against the actual XIs in the FPL data."""
    if not os.path.exists(HIST):
        raise SystemExit("no source_history yet — seed some sources first")
    h = pd.read_csv(HIST)
    h = h[(h.Season.astype(str) == str(season)) & (h.Gameweek == gw)]
    if h.empty:
        raise SystemExit(f"no captured sources for {season} GW{gw}")

    gw_dir = os.path.join(config.FPL_DATA_DIR, "By Gameweek", f"GW{gw}")
    actual = _actual_xis(gw_dir)
    if not actual:
        raise SystemExit(f"actual lineups not available yet for GW{gw} "
                         "(no lineups.csv and no player_gameweek_stats `starts`) — score after it plays")

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
    missing = [s for s in MAIN_SOURCES if s not in set(h.Source)]
    if missing:
        print(f"NOTE: main source(s) not captured for GW{gw}: {', '.join(missing)} "
              "(capture at the deadline — see docs/SOURCE_ACCURACY_CAPTURE.md)")


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
