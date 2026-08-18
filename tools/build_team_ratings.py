"""Odds-implied team strength ratings from football-data.co.uk closing prices.

    python tools/build_team_ratings.py [--seasons 2122 2223 2324 2425 2526]

Writes inputs/team_ratings.csv — one rating per (season, team), where the rating is the
team's mean de-margined closing win probability across the season, averaged over its home
and away halves so venue advantage cancels.

Why this rather than season outright odds: a rating averages 38 closing prices, so it is a
far more precise instrument than three skewed outright markets. Validated 2026-08-14 —
last season's rating predicts THIS season's opening-gameweek match odds at 0.0425 MAE
(0.92-0.97 correlation) across four season transitions, better than the incumbent
win_pred_f3plus manages mid-season. It is however STALE: it cannot know about transfers or
managerial change, which is what the outright odds are for. Blending both beat either
alone by ~20% under cross-validation, so this file is an input to that blend, not a
replacement for the season odds.

Promoted teams have no prior top-flight rating. They inherit the LEVEL of the previous
season's three lowest-RATED teams (not the three relegated — a well-rated side can go
down; West Ham rated 0.275 in 2025-26 and was still relegated, and including it inflates
the promoted level: 0.0449 vs 0.0472 MAE). Ordering among the promoted three does not
matter (0.0448 rank-matched vs 0.0449 flat mean), so they all get the mean. Measured bias
of this substitution is +0.011 on a scale where teams span 0.16-0.60.
"""
import argparse
import io
import os
import sys

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config, names  # noqa: E402

URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
CACHE = os.path.join(config.ROOT, "fpl_data", "football_data")
OUT = os.path.join(config.INPUTS_DIR, "team_ratings.csv")
DEFAULT_SEASONS = ["2122", "2223", "2324", "2425", "2526"]

# football-data's team spellings that differ from ours. Everything else matches; the
# reconciliation printout at the end flags any newcomer so this stays honest.
FOOTBALL_DATA_NAMES = {
    "Man United": "Man Utd", "Tottenham": "Spurs", "Sheffield United": "Sheffield Utd",
    "Ipswich": "Ipswich Town", "Coventry": "Coventry City", "Hull": "Hull City",
    "Nott'm Forest": "Nott'm Forest", "Newcastle": "Newcastle", "Leeds": "Leeds",
}


def season_label(yy):
    return f"20{yy[:2]}-20{yy[2:]}"


def fetch(yy, refresh=False):
    """Season CSV, cached on disk — the site should not be hit on every run."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"E0_{yy}.csv")
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path)
    r = requests.get(URL.format(season=yy), timeout=60)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")))
    d.to_csv(path, index=False)
    print(f"  downloaded {season_label(yy)} -> {os.path.relpath(path, config.ROOT)}")
    return d


def match_probs(d):
    """De-margined closing 1X2. Closing prices are the sharpest the market produces."""
    cols = ["HomeTeam", "AwayTeam", "AvgCH", "AvgCD", "AvgCA"]
    missing = [c for c in cols if c not in d.columns]
    if missing:
        raise SystemExit(f"football-data columns missing: {missing}")
    d = d[cols].dropna().copy()
    overround = 1 / d["AvgCH"] + 1 / d["AvgCD"] + 1 / d["AvgCA"]
    d["p_home"] = (1 / d["AvgCH"]) / overround
    d["p_away"] = (1 / d["AvgCA"]) / overround
    d["HomeTeam"] = names.apply_team_names(d["HomeTeam"].replace(FOOTBALL_DATA_NAMES))
    d["AwayTeam"] = names.apply_team_names(d["AwayTeam"].replace(FOOTBALL_DATA_NAMES))
    return d


def rate(d):
    """Mean win probability at home and away, averaged so venue advantage cancels."""
    home = d.groupby("HomeTeam")["p_home"].mean()
    away = d.groupby("AwayTeam")["p_away"].mean()
    return ((home + away) / 2).rename("rating").sort_values(ascending=False)


def promoted_level(prev_ratings):
    """The level a newly promoted side is assumed to arrive at."""
    return float(prev_ratings.nsmallest(3).mean())


def build(seasons, refresh=False):
    ratings, rows = {}, []
    for yy in seasons:
        r = rate(match_probs(fetch(yy, refresh)))
        ratings[yy] = r
        for team, value in r.items():
            rows.append({"Season": season_label(yy), "Team": team, "rating": round(value, 6),
                         "source": "matches"})

    # Every season after the first gets an explicit promoted-team level for the NEXT one
    for prev, cur in zip(seasons, seasons[1:]):
        level = promoted_level(ratings[prev])
        newcomers = [t for t in ratings[cur].index if t not in ratings[prev].index]
        print(f"  {season_label(cur)}: promoted {', '.join(newcomers) or '(none)'} "
              f"-> assumed level {level:.3f}")

    return pd.DataFrame(rows), ratings


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--refresh", action="store_true", help="re-download instead of using the cache")
    args = ap.parse_args()

    print("Building odds-implied team ratings from football-data closing prices")
    table, ratings = build(args.seasons, args.refresh)

    # Project the upcoming season: last season's ratings, with promoted sides substituted
    last = args.seasons[-1]
    level = promoted_level(ratings[last])
    known = set(ratings[last].index)
    current = set(pd.read_csv(os.path.join(config.INPUTS_DIR, "season_fixtures.csv"))["home_team"])
    promoted = sorted(current - known)
    stale = sorted(known - current)

    for team in promoted:
        table.loc[len(table)] = {"Season": config.SEASON, "Team": team,
                                 "rating": round(level, 6), "source": "promoted-substitute"}
    for team in sorted(current & known):
        table.loc[len(table)] = {"Season": config.SEASON, "Team": team,
                                 "rating": round(float(ratings[last][team]), 6),
                                 "source": "carried-forward"}

    table.to_csv(OUT, index=False)
    print(f"\n{config.SEASON}: {len(promoted)} promoted at {level:.3f} "
          f"({', '.join(promoted)}); {len(stale)} relegated out ({', '.join(stale)})")
    print(f"wrote {len(table)} rows -> {os.path.relpath(OUT, config.ROOT)}")

    cur = table[table["Season"] == config.SEASON].sort_values("rating", ascending=False)
    print(f"\n{config.SEASON} ratings:")
    for r in cur.itertuples():
        tag = "  <- promoted" if r.source == "promoted-substitute" else ""
        print(f"  {r.Team:<16}{r.rating:.3f}{tag}")
