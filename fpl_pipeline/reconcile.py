"""Name reconciliation: detect names that fail to join across data sources.

Every join in the pipeline is an exact-match lookup (workbook VLOOKUP semantics), so a
name mismatch silently degrades output — a lineup player who doesn't match the roster
gets start probability 0, a player missing from the odds tables contributes no points.
This module makes those failures loud: it reports every unmatched name, classifies it,
and where possible suggests the `inputs/name_mappings.csv` row that would fix it.

Runs as a pipeline stage (improved mode) and standalone:
    python -m fpl_pipeline.reconcile
"""
import unicodedata
from difflib import get_close_matches

import pandas as pd
from fpl_pipeline import names

PLAYER_MARKETS = ("score1", "score2", "assist", "yellow")
TEAM_MARKETS = ("clean_sheet", "concede", "gk_saves", "f2_clean_sheet", "f2_concede")


def _norm(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().casefold().strip()


def _is_combo_market(name):
    """Bookmaker combo selections ('A or B to assist a goal') never match a player."""
    low = str(name).lower()
    return " or " in low or " and " in low or " & " in low


def _forenames_compatible(a, b):
    """Guard the surname rule against same-surname DIFFERENT players.

    A unique surname match is not enough on its own: it once proposed
    "Jair Cunha -> Matheus Cunha", a Forest defender onto a Man Utd forward, which would
    have piped one player's goal and assist odds into another's projection with no error
    and no visible symptom. Require the forenames to be plausibly the same person —
    identical, one a shortening of the other (Nico/Nicolas), or a shared stem
    (Andy/Andrew) — so genuine variants still resolve while collisions are refused.
    """
    fa, fb = a.split()[0] if a.split() else "", b.split()[0] if b.split() else ""
    if not fa or not fb:
        return False
    return fa == fb or fa.startswith(fb) or fb.startswith(fa) or fa[:3] == fb[:3]


def _suggest(name, roster_names, norm_map, team_of=None, allowed=None):
    """Best-effort roster match for an unmatched name: accent/case fold, then a token match
    in ANY position, then fuzzy. Returns (suggestion, how) or (None, None).

    The token stage matches on tokens wherever they sit, not just the last one — the old
    last-token rule missed names whose distinguishing token is not final ('Bruno Guimaraes'
    vs roster 'Bruno Guimaraes Rodrigues da Silva'; 'Brau Blanquez' vs 'Brau X Blanquez').
    Two ways in, each requiring a UNIQUE hit in the pool:
      - abbrev:  one name's tokens are a subset of the other's (an abbreviation) — strong on
                 its own, uniqueness handles same-surname ties ('Silva' -> two Silvas = skip).
      - surname: they share a NON-forename token (a surname in any position) AND the forenames
                 are compatible — the guard that refuses 'Jair Cunha' -> 'Matheus Cunha'.
    """
    n = _norm(name)
    if n in norm_map:
        return norm_map[n], "accent/case"
    # Restrict to players in the fixture this name was priced in, when we know it
    pool = roster_names
    if allowed and team_of:
        pool = [r for r in roster_names if team_of.get(r) in allowed] or roster_names
    suffix = "+team" if pool is not roster_names else ""
    tn = n.split()
    if tn:
        tn_set = set(tn)
        abbrev, surname = [], []
        for r in pool:
            tr_set = set(_norm(r).split())
            shared = tn_set & tr_set
            if not shared:
                continue
            if tn_set <= tr_set or tr_set <= tn_set:
                abbrev.append(r)                       # one name abbreviates the other
            elif (shared - {tn[0]}) and _forenames_compatible(n, _norm(r)):
                surname.append(r)                      # shared surname (any position), same forename
        for hits, how in ((abbrev, "abbrev"), (surname, "surname")):
            if len(hits) == 1:
                return hits[0], how + suffix
    fuzzy = get_close_matches(name, pool, n=1, cutoff=0.8)
    if fuzzy:
        return fuzzy[0], "fuzzy" + suffix
    return None, None


def _fixture_teams(sportsbet, team_names):
    """player -> the teams that appear in the fixtures they are priced in.

    Raw odds carry `match_id` as "Arsenal vs. Coventry City", so a priced player must
    belong to one of those two sides. That constrains a surname match to the fixture the
    odds came from, which is a far tighter filter than name shape alone.
    """
    out = {}
    for df in (sportsbet or {}).values():
        if df is None or getattr(df, "empty", True):
            continue
        if "player_name" not in df.columns or "match_id" not in df.columns:
            continue
        for player, match in zip(df["player_name"], df["match_id"]):
            sides = set()
            for sep in (" vs. ", " vs ", " v "):
                if sep in str(match):
                    sides = {s.strip() for s in str(match).split(sep, 1)}
                    break
            if sides:
                out.setdefault(player, set()).update(
                    names.apply_team_names(pd.Series(sorted(sides))))
    return out


def report(roster, lineups, mkts, sportsbet=None):
    """Reconciliation rows: (source, name, issue, suggestion, how). Empty = all clean."""
    roster_names = list(roster["name"])
    roster_set = set(roster_names)
    norm_map = {_norm(n): n for n in roster_names}
    roster_teams = set(roster["team"])
    team_of = dict(zip(roster["name"], roster["team"]))
    in_fixture = _fixture_teams(sportsbet, roster_teams)
    rows = []

    def add(source, name, issue):
        suggestion, how = _suggest(name, roster_names, norm_map,
                                   team_of=team_of, allowed=in_fixture.get(name))
        rows.append({"source": source, "name": name, "issue": issue,
                     "suggestion": suggestion, "how": how})

    # 1) Starting lineups vs roster — critical: unmatched players get start prob 0
    for p in sorted(set(lineups["Player"]) - roster_set):
        add("starting_lineups", p, "lineup player not in FPL roster (start prob lost)")
    for t in sorted(set(lineups["Team"]) - roster_teams):
        add("starting_lineups", t, "lineup team not an FPL team name")

    # 2) Player-market odds vs roster — these odds can never reach a player
    for key in PLAYER_MARKETS:
        if key not in mkts or "player" not in mkts[key].columns:
            continue
        for p in sorted(set(mkts[key]["player"].dropna()) - roster_set):
            if _is_combo_market(p):
                continue  # structurally unmatchable, not a naming problem
            add(f"odds:{key}", p, "odds player not in FPL roster (odds unusable)")

    # 3) XI players with no attacking odds at all — the join worked, the data is absent
    starters = lineups[pd.to_numeric(lineups["F1"], errors="coerce").fillna(0) > 0]
    positions = roster.set_index("name")["position"].astype(str)
    with_score = set(mkts.get("score1", pd.DataFrame(columns=["player"]))["player"])
    with_assist = set(mkts.get("assist", pd.DataFrame(columns=["player"]))["player"])
    for p in sorted(set(starters["Player"]) & roster_set):
        if positions.get(p) != "GK" and p not in with_score and p not in with_assist:
            rows.append({"source": "coverage", "name": p,
                         "issue": "XI starter with no score/assist odds (0 attacking XP)",
                         "suggestion": None, "how": None})

    # 4) Team-market keys vs FPL team names
    for key in TEAM_MARKETS:
        if key not in mkts or "team" not in mkts[key].columns:
            continue
        for t in sorted(set(mkts[key]["team"].dropna()) - roster_teams):
            rows.append({"source": f"odds:{key}", "name": t,
                         "issue": "odds team not an FPL team name (lookups miss)",
                         "suggestion": None, "how": None})

    return pd.DataFrame(rows, columns=["source", "name", "issue", "suggestion", "how"])


def print_summary(rec):
    if rec.empty:
        print("  name reconciliation: all sources join cleanly")
        return
    print(f"  name reconciliation: {len(rec)} issues "
          f"({rec['source'].value_counts().to_dict()})")
    fixable = rec[rec["suggestion"].notna()]
    if len(fixable):
        print("  suggested inputs/name_mappings.csv rows (verify before adding):")
        for _, r in fixable.iterrows():
            print(f"    player,{r['name']},{r['suggestion']}   # {r['source']}, {r['how']}")
    for _, r in rec[rec["suggestion"].isna()].head(15).iterrows():
        print(f"    UNRESOLVED [{r['source']}] {r['name']}: {r['issue']}")


if __name__ == "__main__":
    from . import ingest, markets, names

    inputs = ingest.load_inputs()
    inputs["starting_lineups"]["Player"] = names.apply_player_names(
        inputs["starting_lineups"]["Player"])
    sportsbet = ingest.load_sportsbet()
    rec = report(ingest.load_fpl_players(), inputs["starting_lineups"],
                 markets.build_all(sportsbet, inputs, dedup_f2=True))
    print_summary(rec)
    if not rec.empty:
        rec.to_csv("outputs/name_reconciliation.csv", index=False)
        print("  full report -> outputs/name_reconciliation.csv")
