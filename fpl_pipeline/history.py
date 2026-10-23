"""Historical archives: automatic upserts that replace the old manual paste-values ritual.

- Player history (inputs/historical_player_data.csv): one full F1-F8 snapshot per player
  per gameweek, keyed by the Gameweek column — re-running the same gameweek replaces its
  rows instead of appending duplicates.
  NOTE: the F2-F8 columns (Win Pred/Score/Assist/Opponent/Venue/...) are the PREDICTIONS we
  made that week for future fixtures — keep them. They are the record used to BACK-TEST the
  forward projections: score a gameweek-M F{k} prediction against the real odds-derived F1
  value archived when that fixture became current (gameweek M+k-1), the F{k} Opponent/Venue
  identifying it as the same fixture. The rate/factor code only reads the F1 block, but the
  F2-F8 columns are the substrate for evaluating and improving the projection models, so a
  cleanup pass must NOT drop them as "unused" (see tools/backtest_projections.py). The values
  are snapshotted from the master, so anything the master carries populates automatically.
- Fixture history (inputs/historical_fixture_odds.csv): one row per fixture with match
  and season odds, keyed by the (home_team, away_team) pair.
- Fallback factors (inputs/fallback_factors.csv): per-player factors refreshed from the
  latest computed values; only non-NaN factors overwrite existing entries.
- Raw bookmaker prices (inputs/raw_odds/<season>/GW<NN>/): every sportsbet file the provenance
  manifest marks REAL, as scraped (synthetic seed rows dropped), taken at each --gw run. The player archive stores our adjusted
  chances (de-margined + longshot curve); these keep the untouched prices so any curve can be re-fit
  from scratch later.

All functions take explicit paths so tests can run against temporary files.
"""
import os

import numpy as np
import pandas as pd

from . import config

PLAYER_HISTORY_CSV = os.path.join(config.INPUTS_DIR, "historical_player_data.csv")
FIXTURE_HISTORY_CSV = os.path.join(config.INPUTS_DIR, "historical_fixture_odds.csv")
FALLBACK_FACTORS_CSV = os.path.join(config.INPUTS_DIR, "fallback_factors.csv")
RAW_ODDS_DIR = os.path.join(config.INPUTS_DIR, "raw_odds")

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


def withheld_columns(columns, withhold):
    """Columns whose name contains any of the `withhold` fragments (e.g. 'Yellow Card' -> F1 Yellow
    Card, F1 Yellow Card Factor, F2..F8 Yellow Card). Used to blank a synthetic market's columns."""
    return [c for c in columns if any(frag in c for frag in withhold)]


def update_player_history(master, gameweek, path=PLAYER_HISTORY_CSV, season=None, withhold=()):
    """Upsert this gameweek's F1 block into the player history archive, keyed by
    (Season, Gameweek) so gameweek numbers can recur across seasons.

    withhold: name fragments of markets that are NOT real this week (see run.guard_synthetic_archive);
    their columns are written as NaN so a synthetic carry never enters the factor history."""
    season = season or config.SEASON
    hist = pd.read_csv(path, low_memory=False)
    hist = hist.loc[:, ~hist.columns.str.startswith("Unnamed")]  # legacy sheet padding

    snapshot_cols = [c for c in hist.columns if c not in ("Season", "Gameweek")]
    missing = [c for c in snapshot_cols if c not in master.columns]
    if missing:
        raise ValueError(f"master is missing history columns: {missing}")

    snapshot = master[snapshot_cols].copy()
    blanked = withheld_columns(snapshot.columns, withhold)
    if blanked:
        snapshot[blanked] = np.nan
        print(f"  history: {len(blanked)} columns withheld as synthetic ({', '.join(withhold)})")
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


def update_fixture_history(wdw, season_probs, path=FIXTURE_HISTORY_CSV, season=None,
                           gameweek=None, sportsbet=None, fixtures=None):
    """Upsert the upcoming gameweek's fixtures (first 10 scraped matches) with their
    match odds and both teams' season odds, keyed by (Season, home_team, away_team).

    Records more than the pipeline currently consumes, because odds are unbackfillable:
      - draw odds, so the 1X2 book can be de-margined properly later (win-only prices
        force an assumed overround, which biases any fitted target);
      - the Gameweek, so a row is dated - "state at M vs odds at N" work needs it;
      - per-team clean-sheet and over-1.5/over-3.5 team-goal odds, which together pin
        each side's implied goal expectation (lambda) - the quantity the clean sheet,
        concede and team-goal markets all ultimately want.
    See task #19: none of this can be reconstructed after the gameweek passes.
    """
    season = season or config.SEASON
    hist = pd.read_csv(path)
    season_idx = season_probs.set_index("team")

    f1 = wdw.iloc[:10]
    if fixtures is not None and gameweek is not None:
        # The upsert key is (home, away), and the Gameweek stamp is whatever --gw said - so a run
        # against last week's W/D/W file would relabel those fixtures (GW2 archived as GW3 on
        # 2026-08-31). Refuse unless the F1 block is the fixtures.csv window for this gameweek.
        col = f"GW{gameweek} Opponent"
        if col not in fixtures.columns:
            print(f"  history: fixtures.csv has no '{col}' column - fixture history NOT archived "
                  f"(roll the window to GW{gameweek} first)")
            return hist
        expected = {frozenset((t, o)) for t, o in zip(fixtures.iloc[:, 0], fixtures[col])
                    if isinstance(o, str)}
        got = [frozenset((h, a)) for h, a in zip(f1.iloc[:, 0], f1.iloc[:, 1])]
        overlap = sum(p in expected for p in got) / max(len(got), 1)
        if overlap < 0.7:
            print(f"  history: W/D/W F1 block matches only {overlap:.0%} of the GW{gameweek} fixtures "
                  f"- fixture history NOT archived (stale odds file? re-scrape)")
            return hist
    rows = pd.DataFrame({
        "Season": season,
        "Gameweek": gameweek,
        "home_team": f1.iloc[:, 0].values,
        "away_team": f1.iloc[:, 1].values,
        "home_win_odds": f1.iloc[:, 2].values,
        "away_win_odds": f1.iloc[:, 3].values,
        "draw_odds": f1["draw_odds"].values if "draw_odds" in f1.columns else pd.NA,
    })

    # Per-team markets are keyed by team name within the same gameweek's scrape
    cs = tg = None
    if sportsbet is not None:
        cs = sportsbet.get("clean_sheet")
        tg = sportsbet.get("team_goals")
        cs = cs.drop_duplicates(subset="team_name").set_index("team_name") if cs is not None else None
        tg = tg.drop_duplicates(subset="Team").set_index("Team") if tg is not None else None

    for side in ("home", "away"):
        teams = rows[f"{side}_team"]
        rows[f"{side}_title_odds"] = teams.map(season_idx["title_odds"]).values
        rows[f"{side}_relegation_odds"] = teams.map(season_idx["relegation_odds"]).values
        rows[f"{side}_top_6_odds"] = teams.map(season_idx["top6_odds"]).values
        rows[f"{side}_clean_sheet_odds"] = (
            teams.map(cs["clean_sheet_yes"]).values if cs is not None else pd.NA)
        for line in ("1.5", "3.5"):
            rows[f"{side}_over_{line}_odds"] = (
                teams.map(tg[f"Team_Over_{line}"]).values if tg is not None else pd.NA)

    # Columns added after earlier seasons were recorded: backfill the archive as NA
    for col in rows.columns:
        if col not in hist.columns:
            hist[col] = pd.NA

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
    frame = archived_factor_rows(stat, season=season or config.SEASON, path=path)
    return {p: g["factor"].to_numpy() for p, g in frame.groupby("player")}


# fallback-factor rebuild (2026-10-08): the fallback sheet used to be overwritten each --gw run with
# whatever the live master computed. Synthetic-seeded weeks leaked placeholder factors into it
# (build_synthetic_gw floors unpriced players at a 1% price, and the withhold guard is per MARKET,
# not per player) — Joelinton's assist factor sat at 0.01, Mitoma's score factor at 0.015, 59 rows
# below 0.05. The archive is clean (it stores real-odds weeks only; no archived score/assist factor
# is below 0.1), so the sheet is now REBUILT from it: each player's median over his most recent
# real weeks — the same median the forward projection already trusts.
FALLBACK_RECENT_WEEKS = 10
FALLBACK_MIN_FACTOR = 0.05      # below this a factor is a placeholder, never a real price
FACTOR_STATS = {
    "Score 1+ Factor": "score1", "Assist Factor": "assist", "F1 Yellow Card Factor": "yellow",
    "F1 Concede 2+ Goals Factor": "concede2", "F1 Concede 4+ Goals Factor": "concede4",
    "F1 3+ Saves Factor": "saves3", "F1 6+ Saves Factor": "saves6",
}


def archived_factor_rows(stat="score1", season=None, path=PLAYER_HISTORY_CSV):
    """One row per archived (player, week) odds factor for `stat`: columns player, Season,
    Gameweek, factor. `season=None` reads every season. Factors are recomputed from the archived
    probability / baseline (assists brought onto today's calibration curve), exactly as the
    trailing-median forward projection does, so both always agree."""
    from . import model

    empty = pd.DataFrame(columns=["player", "Season", "Gameweek", "factor"])
    if not os.path.exists(path):
        return empty
    hist = pd.read_csv(path, low_memory=False)
    if season is not None:
        hist = hist[hist["Season"] == season]
    if hist.empty:
        return empty
    prob_col = {"score1": "F1 Score 1+", "assist": "F1 Assist",
                "yellow": "F1 Yellow Card", "clean_sheet": "F1 Clean Sheet",
                "concede2": "F1 Concede 2+ Goals", "concede4": "F1 Concede 4+ Goals",
                "saves3": "F1 3+ Saves", "saves6": "F1 6+ Saves"}[stat]
    if not {"F1 Win", "F1 Opponent Win", "Position", "F1 Venue", prob_col} <= set(hist.columns):
        return empty
    for c in ("F1 Win", "F1 Opponent Win", prob_col):
        hist[c] = pd.to_numeric(hist[c], errors="coerce")
    if stat == "assist":
        # archived weeks were calibrated under whatever assist curve was live then - bring them onto
        # today's curve so the trailing median is not anchored to a retired regime (config.ASSIST_CURVE_REGIMES)
        from . import markets
        gw = pd.to_numeric(hist["Gameweek"], errors="coerce")
        for reg_season, lo, hi, knots in getattr(config, "ASSIST_CURVE_REGIMES", []):
            rows = (hist["Season"] == reg_season) & gw.between(lo, hi)
            if rows.any():
                hist.loc[rows, prob_col] = markets.recalibrate(hist.loc[rows, prob_col], knots,
                                                               config.LONGSHOT_CALIBRATION_ASSIST)
    f = hist[prob_col] / model.baseline(stat, hist["F1 Win"], hist["F1 Opponent Win"],
                                        hist["Position"], hist["F1 Venue"] == "H")
    frame = pd.DataFrame({"player": hist["Player Name"], "Season": hist["Season"],
                          "Gameweek": pd.to_numeric(hist["Gameweek"], errors="coerce"),
                          "factor": f}).dropna()
    # A zero factor means P(stat) = 0 - never a real price (it was the IFERROR(...,0) concede-4+
    # artefact); it would pin the median at 0, so it is not history.
    return frame[frame["factor"] > 0]


def recent_factor_medians(path=PLAYER_HISTORY_CSV, weeks=FALLBACK_RECENT_WEEKS,
                          min_factor=FALLBACK_MIN_FACTOR):
    """Per-player median of each factor over his most recent `weeks` archived real weeks (all
    seasons, newest first). Index = player name, columns = FACTOR_COLUMNS; NaN where a player has
    no usable history for that stat."""
    out = {}
    for col, stat in FACTOR_STATS.items():
        rows = archived_factor_rows(stat, season=None, path=path)
        rows = rows[rows["factor"] >= min_factor].sort_values(["Season", "Gameweek"])
        out[col] = rows.groupby("player")["factor"].apply(lambda s: float(np.median(s.tail(weeks))))
    return pd.DataFrame(out).reindex(columns=FACTOR_COLUMNS)


def refresh_fallback_factors(master=None, path=FALLBACK_FACTORS_CSV, withhold=(),
                             history_path=PLAYER_HISTORY_CSV, weeks=FALLBACK_RECENT_WEEKS,
                             min_factor=FALLBACK_MIN_FACTOR):
    """Rebuild the fallback sheet from the archive (call it after update_player_history).

    - A player with archived real weeks gets the median of his last `weeks` (all seasons).
    - A player with no usable history keeps his stored value — unless it is a placeholder
      (< min_factor), which becomes his position's median instead.
    `withhold` is accepted for call compatibility only: the archive already withholds synthetic
    markets, so nothing synthetic can reach the rebuild. `master` (optional) supplies positions."""
    existing = pd.read_csv(path).drop_duplicates(subset="Player Name").set_index("Player Name")
    existing = existing.reindex(columns=FACTOR_COLUMNS)
    rebuilt = recent_factor_medians(history_path, weeks, min_factor)

    out = existing.copy()
    out.update(rebuilt)                                   # history wins wherever it exists
    new_players = rebuilt.loc[~rebuilt.index.isin(out.index)]
    out = pd.concat([out, new_players])

    if master is not None and {"Player Name", "Position"} <= set(master.columns):
        pos = master.drop_duplicates("Player Name").set_index("Player Name")["Position"]
    else:
        hist = pd.read_csv(history_path, low_memory=False) if os.path.exists(history_path) else pd.DataFrame()
        pos = (hist.drop_duplicates("Player Name", keep="last").set_index("Player Name")["Position"]
               if {"Player Name", "Position"} <= set(hist.columns) else pd.Series(dtype=object))
    position = out.index.to_series().map(pos)

    repaired = 0
    for col in FACTOR_COLUMNS:
        bad = out[col] < min_factor                       # placeholder with no history to replace it
        if not bad.any():
            continue
        good = out[col].where(out[col] >= min_factor)
        by_pos = good.groupby(position).median()
        fill = position[bad].map(by_pos).fillna(good.median())
        out.loc[bad, col] = fill
        repaired += int(bad.sum())

    out.index.name = "Player Name"
    out = out.reset_index()
    out.to_csv(path, index=False)
    print(f"  history: fallback factors rebuilt from the archive ({len(rebuilt)} players with history, "
          f"{len(new_players)} new, {repaired} placeholder values -> position median) -> {os.path.basename(path)}")
    return out


# Seed rows written by build_synthetic_gw carry NUMERIC match ids / dates (90000000+idx, epoch
# stamps); Betway rows carry fixture labels and ISO dates. A file marked "real" is real per MATCH, not
# per row - betway.write keeps the seed for fixtures it didn't price - so those rows are dropped here.
_SEED_MARKER_COLUMNS = ("match_id", "date", "Date")


def _drop_seed_rows(frame):
    col = next((c for c in _SEED_MARKER_COLUMNS if c in frame.columns), None)
    if col is None:          # win/draw/win carries no marker; Betway prices every fixture it lists
        return frame
    return frame[pd.to_numeric(frame[col], errors="coerce").isna()]


def archive_raw_odds(season, gameweek, src_dir=None, dest_root=None):
    """Save this week's REAL bookmaker prices to <dest_root>/<season>/GW<NN>/, one file per market,
    exactly as scraped (read as text, so no number is reformatted) minus any synthetic seed rows.
    Markets that are synthetic or derived (computed by us from real prices) are skipped. A rerun of
    the same gameweek overwrites file by file, so the deadline run leaves the deadline prices; a
    market saved earlier is never deleted. The matching slice of the provenance manifest is kept
    alongside as _provenance.json. Returns {filename: rows saved}."""
    import json

    from . import provenance

    src_dir = src_dir or config.SPORTSBET_DIR
    dest = os.path.join(dest_root or RAW_ODDS_DIR, season, f"GW{int(gameweek):02d}")
    markets = provenance.status().get("markets", {})
    saved = {}
    for f in sorted(markets):
        if markets[f].get("state") != "real" or not os.path.exists(os.path.join(src_dir, f)):
            continue
        rows = _drop_seed_rows(pd.read_csv(os.path.join(src_dir, f), dtype=str, keep_default_na=False))
        if rows.empty:
            continue
        os.makedirs(dest, exist_ok=True)
        rows.to_csv(os.path.join(dest, f), index=False)
        saved[f] = len(rows)
    if saved:
        manifest = os.path.join(dest, "_provenance.json")
        doc = {}
        if os.path.exists(manifest):
            with open(manifest, encoding="utf-8") as fh:
                doc = json.load(fh)
        doc.update({f: markets[f] for f in saved})
        with open(manifest, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
    return saved
