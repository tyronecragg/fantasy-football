"""Historical archives: automatic upserts that replace the old manual paste-values ritual.

- Player history (inputs/historical_player_data.csv): one F1-block snapshot per player
  per gameweek, keyed by the Gameweek column — re-running the same gameweek replaces its
  rows instead of appending duplicates.
- Fixture history (inputs/historical_fixture_odds.csv): one row per fixture with match
  and season odds, keyed by the (home_team, away_team) pair.
- Fallback factors (inputs/fallback_factors.csv): per-player factors refreshed from the
  latest computed values; only non-NaN factors overwrite existing entries.

All functions take explicit paths so tests can run against temporary files.
"""
import os

import pandas as pd

from . import config

PLAYER_HISTORY_CSV = os.path.join(config.INPUTS_DIR, "historical_player_data.csv")
FIXTURE_HISTORY_CSV = os.path.join(config.INPUTS_DIR, "historical_fixture_odds.csv")
FALLBACK_FACTORS_CSV = os.path.join(config.INPUTS_DIR, "fallback_factors.csv")

FACTOR_COLUMNS = [
    "Score 1+ Factor", "Assist Factor", "F1 Yellow Card Factor",
    "F1 Concede 2+ Goals Factor", "F1 Concede 4+ Goals Factor",
    "F1 3+ Saves Factor", "F1 6+ Saves Factor",
]


def infer_gameweek():
    """Upcoming gameweek = last played gameweek + 1, from the FPL data. None when it
    can't be determined confidently (then the caller should require --gw)."""
    try:
        stats = pd.read_csv(os.path.join(config.FPL_DATA_DIR, "playerstats.csv"))
        gw = int(pd.to_numeric(stats["gw"], errors="coerce").max()) + 1
    except (FileNotFoundError, KeyError, ValueError):
        return None
    return gw if 1 <= gw <= 38 else None


def update_player_history(master, gameweek, path=PLAYER_HISTORY_CSV, season=None):
    """Upsert this gameweek's F1 block into the player history archive, keyed by
    (Season, Gameweek) so gameweek numbers can recur across seasons."""
    season = season or config.SEASON
    hist = pd.read_csv(path, low_memory=False)
    hist = hist.loc[:, ~hist.columns.str.startswith("Unnamed")]  # legacy sheet padding

    snapshot_cols = [c for c in hist.columns if c not in ("Season", "Gameweek")]
    missing = [c for c in snapshot_cols if c not in master.columns]
    if missing:
        raise ValueError(f"master is missing history columns: {missing}")

    snapshot = master[snapshot_cols].copy()
    snapshot.insert(0, "Gameweek", gameweek)
    snapshot.insert(0, "Season", season)

    existing_gw = pd.to_numeric(hist["Gameweek"], errors="coerce")
    same = (hist["Season"] == season) & (existing_gw == gameweek)
    replaced = int(same.sum())
    hist = pd.concat([hist[~same], snapshot], ignore_index=True)
    hist.to_csv(path, index=False)

    action = f"replaced {replaced} rows" if replaced else "appended"
    print(f"  history: {season} GW{gameweek} player snapshot ({len(snapshot)} rows, {action}) -> {os.path.basename(path)}")
    return hist


def update_fixture_history(wdw, season_probs, path=FIXTURE_HISTORY_CSV, season=None):
    """Upsert the upcoming gameweek's fixtures (first 10 scraped matches) with their
    match odds and both teams' season odds, keyed by (Season, home_team, away_team)."""
    season = season or config.SEASON
    hist = pd.read_csv(path)
    season_idx = season_probs.set_index("team")

    f1 = wdw.iloc[:10]
    rows = pd.DataFrame({
        "Season": season,
        "home_team": f1.iloc[:, 0].values,
        "away_team": f1.iloc[:, 1].values,
        "home_win_odds": f1.iloc[:, 2].values,
        "away_win_odds": f1.iloc[:, 3].values,
    })
    for side in ("home", "away"):
        teams = rows[f"{side}_team"]
        rows[f"{side}_title_odds"] = teams.map(season_idx["title_odds"]).values
        rows[f"{side}_relegation_odds"] = teams.map(season_idx["relegation_odds"]).values
        rows[f"{side}_top_6_odds"] = teams.map(season_idx["top6_odds"]).values

    key_cols = ["Season", "home_team", "away_team"]
    new_keys = set(map(tuple, rows[key_cols].values))
    is_replaced = hist[key_cols].apply(tuple, axis=1).isin(new_keys)
    hist = pd.concat([hist[~is_replaced], rows[hist.columns]], ignore_index=True)
    hist.to_csv(path, index=False)

    print(f"  history: {len(rows)} fixtures upserted ({int(is_replaced.sum())} replaced) -> {os.path.basename(path)}")
    return hist


def season_weekly_factors(stat="score1", season=None, path=PLAYER_HISTORY_CSV):
    """Per-player arrays of this season's weekly odds factors for `stat`, from the
    archive. Feeds the trailing-median factor (factor experiment 2026-08: median
    improved score-projection holdout MAE ~5%; assists were WORSE with median, so
    only score uses this). Empty dict early in a season — behaviour then reduces to
    the single-week factor."""
    from . import model

    season = season or config.SEASON
    if not os.path.exists(path):
        return {}
    hist = pd.read_csv(path, low_memory=False)
    hist = hist[hist["Season"] == season]
    if hist.empty:
        return {}
    prob_col = {"score1": "F1 Score 1+", "assist": "F1 Assist",
                "yellow": "F1 Yellow Card", "clean_sheet": "F1 Clean Sheet",
                "concede2": "F1 Concede 2+ Goals", "concede4": "F1 Concede 4+ Goals",
                "saves3": "F1 3+ Saves", "saves6": "F1 6+ Saves"}[stat]
    for c in ("F1 Win", "F1 Opponent Win", prob_col):
        hist[c] = pd.to_numeric(hist[c], errors="coerce")
    f = hist[prob_col] / model.baseline(stat, hist["F1 Win"], hist["F1 Opponent Win"],
                                        hist["Position"], hist["F1 Venue"] == "H")
    frame = pd.DataFrame({"player": hist["Player Name"], "factor": f}).dropna()
    return {p: g["factor"].to_numpy() for p, g in frame.groupby("player")}


def refresh_fallback_factors(master, path=FALLBACK_FACTORS_CSV):
    """Update stored per-player factors from the latest run. Only players with F1 odds
    contribute, and only their non-NaN factors overwrite existing values, so a player
    with a missing market this week keeps last week's factor."""
    existing = pd.read_csv(path).drop_duplicates(subset="Player Name").set_index("Player Name")

    fresh = (master.loc[master["F1 Win"].notna(), ["Player Name"] + FACTOR_COLUMNS]
             .drop_duplicates(subset="Player Name").set_index("Player Name"))

    existing.update(fresh)  # overwrites only where fresh is non-NaN
    new_players = fresh.loc[~fresh.index.isin(existing.index)]
    out = pd.concat([existing, new_players]).reset_index()
    out.to_csv(path, index=False)

    print(f"  history: fallback factors refreshed ({len(fresh)} players, {len(new_players)} new) -> {os.path.basename(path)}")
    return out
