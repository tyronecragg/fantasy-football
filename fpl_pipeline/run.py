"""Run the full pipeline with a CSV snapshot per stage.

Usage:  python -m fpl_pipeline.run [--validate] [--gw N]

Default (improved) mode applies the model improvements documented in players.py.
Passing --gw N additionally upserts the historical archives and refreshes the fallback
factors in inputs/ (re-running the same gameweek replaces its rows). Without --gw the
archives are never touched — inference from FPL data proved unreliable and is only
printed as a hint.

--validate runs in parity mode: improvements and input-mutating side effects are
disabled and the output is compared cell-for-cell with the workbook. If a coefficient
refit has replaced coefficients.json, parity automatically uses the backed-up
coefficients_workbook.json.
"""
import os
import sys

import pandas as pd

from . import (config, curation_check, history, ingest, markets, model, names, players,
               provenance, rating as rating_mod, reconcile, team_model)
from .io_utils import reset_counter, snapshot


# Scraped markets that feed the per-player factors (player props AND the per-team clean-sheet /
# team-goals markets, which drive the clean-sheet and concede factors). A market still synthetic
# has its columns withheld from the player archive (see _MARKET_COLUMNS). two_assists is
# runtime-derived from assist, not independently scraped, so it does not gate.
_PLAYER_MARKETS = ["sportsbet_goalscorer_odds.csv", "sportsbet_two_goals_odds.csv",
                   "sportsbet_assist_odds.csv", "sportsbet_booking_odds.csv",
                   "sportsbet_clean_sheet_odds.csv", "sportsbet_team_goals_odds.csv",
                   "sportsbet_goalkeeper_saves_odds.csv"]


# Player-market file -> the master/history column fragments it feeds (every column whose name
# contains the fragment: F1 value, its factor, F2-F8 projections). A market that is not grounded
# (a synthetic carry) has these withheld from the player snapshot and the fallback refresh, so a
# partial scrape archives what IS real instead of nothing - and never needs --force-archive.
_MARKET_COLUMNS = {
    "sportsbet_goalscorer_odds.csv": ("Score 1+", "Score 2+", "Score 3+"),
    "sportsbet_two_goals_odds.csv": ("Score 2+", "Score 3+"),
    "sportsbet_assist_odds.csv": ("Assist",),
    "sportsbet_booking_odds.csv": ("Yellow Card",),
    "sportsbet_clean_sheet_odds.csv": ("Clean Sheet",),
    "sportsbet_team_goals_odds.csv": ("Concede",),
    "sportsbet_goalkeeper_saves_odds.csv": ("Saves",),
}
_WDW = "sportsbet_win_draw_win_odds.csv"


def withheld_player_columns(force=False):
    """Column-name fragments to blank in this week's player archive: those of every player market
    that is not grounded (real or derived-from-real) in the provenance manifest."""
    if force:
        return ()
    return tuple(sorted({frag for f in _PLAYER_MARKETS if not provenance.is_grounded(f)
                         for frag in _MARKET_COLUMNS[f]}))


def guard_synthetic_archive(gameweek, force=False):
    """Decide what may be archived, driven by the odds provenance manifest (fpl_pipeline/provenance.py).

    Returns "all" | "fixtures_only" | "none".
      - "none" when the manifest is stamped for a different gameweek than --gw: the sportsbet files
        are not this week's odds, so archiving them under this gameweek would mislabel the fixture
        archive and record a stale carry as player history (both happened on 2026-08-31).
      - "fixtures_only" when EVERY player market is still the synthetic seed (right after a roll).
      - "all" otherwise. Player markets that are still synthetic don't block the rest: their columns
        are withheld from the snapshot (see withheld_player_columns) while everything grounded -
        real, or derived at scrape time from real prices (2+ goals, GK saves) - is archived.
    The manifest is stamped by the writers (betway=real/derived, synthetic seed and partial card
    scrapes=synthetic), so this self-clears as markets land. `tools/odds_status.py` shows the
    current state; --force-archive records everything regardless.
    """
    if not gameweek:
        return "none"
    if force:
        return "all"
    manifest_gw = provenance.status().get("gw")
    if manifest_gw is not None and int(manifest_gw) != int(gameweek):
        print(f"  Odds provenance is stamped GW{manifest_gw} but this run is --gw {gameweek}: the sportsbet\n"
              f"  files are not this gameweek's odds. NOTHING archived - roll/scrape first, or --force-archive.")
        return "none"
    synthetic = [provenance.FRIENDLY.get(f, f) for f in _PLAYER_MARKETS if not provenance.is_grounded(f)]
    if not synthetic:
        return "all"
    if len(synthetic) == len(_PLAYER_MARKETS):
        print(f"  Every player market is still synthetic (see tools/odds_status.py).\n"
              f"  Archiving MATCH odds only (unbackfillable); player history withheld.")
        return "fixtures_only"
    print(f"  Player markets still synthetic: {', '.join(synthetic)} (see tools/odds_status.py).\n"
          f"  Archiving everything else; their columns ({', '.join(withheld_player_columns())}) are\n"
          f"  withheld from the player snapshot and the fallback-factor refresh.")
    return "all"


def run(parity_mode=False, gameweek=None, force_archive=False):
    archive_mode = guard_synthetic_archive(gameweek, force_archive)
    reset_counter(subdir="parity" if parity_mode else None)
    improved = not parity_mode

    live_data_dir = config.FPL_DATA_DIR
    if parity_mode:
        print(f"Running in parity mode (workbook-exact, improvements disabled, "
              f"season pinned to {config.PARITY_SEASON})")
        config.FPL_DATA_DIR = os.path.join(
            config.ROOT, "fpl_data", "FPL-Core-Insights", "data", config.PARITY_SEASON)
        if os.path.exists(model.WORKBOOK_COEFFICIENTS_JSON):
            print("  using workbook-extracted coefficients (coefficients_workbook.json)")
            model.load_coefficients(model.WORKBOOK_COEFFICIENTS_JSON)
    else:
        model.load_coefficients()

    try:
        return _run(parity_mode, gameweek, improved, archive_mode, force_archive)
    finally:
        config.FPL_DATA_DIR = live_data_dir


def _run(parity_mode, gameweek, improved, archive_mode="none", force_archive=False):
    print("Loading sources...")
    if parity_mode:
        inputs = ingest.load_inputs(config.PARITY_INPUTS_DIR)
        sportsbet = ingest.load_sportsbet(config.PARITY_SPORTSBET_DIR)
    else:
        inputs = ingest.load_inputs()
        # Lineups go through the same name mapper as the roster/odds, so a
        # name_mappings.csv row can fix a lineup spelling (accent/alias) too.
        inputs["starting_lineups"]["Player"] = names.apply_player_names(
            inputs["starting_lineups"]["Player"])
        sportsbet = ingest.load_sportsbet()

    if parity_mode:
        # The workbook's own frozen roster/DC data: upstream FPL data rewrites history
        roster = snapshot(ingest.load_fpl_players_workbook(), "fpl_players")
        dc_stats = ingest.load_defensive_contributions_workbook()
    else:
        roster = snapshot(ingest.load_fpl_players(), "fpl_players")
        dc_stats = ingest.load_defensive_contributions()
    snapshot(dc_stats["DEF"], "dc_stats_def")
    snapshot(dc_stats["MID"], "dc_stats_mid")

    season = snapshot(team_model.season_probs(inputs, workbook_quirks=parity_mode), "season_probs")
    teamview = snapshot(team_model.team_fixture_view(inputs, sportsbet, draw_aware=improved),
                        "team_fixture_view")

    mkts = markets.build_all(sportsbet, inputs, dedup_f2=improved, roster_names=roster["name"])
    for key in ("score1", "score2", "assist", "yellow", "clean_sheet", "concede", "gk_saves"):
        snapshot(mkts[key], f"market_{key}")

    # The master must stay snapshot #13 — the optimisers read outputs/13_players_master.csv
    # by name, so variable-count stages (reconciliation) come after it.
    factor_history = None
    assist_real = provenance.is_real("sportsbet_assist_odds.csv")
    if improved:
        # assist normally stays single-week (backtested better on real weeks), but it needs its
        # archive history available so players.build can fall back to the trailing median when the
        # assist market ISN'T real. Every other stat medians unconditionally (MEDIAN_FACTOR_STATS).
        hist_stats = tuple(config.MEDIAN_FACTOR_STATS) + ("assist",)
        factor_history = {stat: h for stat in hist_stats
                          if (h := history.season_weekly_factors(stat))}
        if factor_history:
            print(f"  trailing-median factors active for {len(factor_history)} stats "
                  f"({len(next(iter(factor_history.values())))} players with archive weeks)"
                  + ("" if assist_real else "; assist market not real -> assist on trailing median"))

    rating = None
    if improved and config.USE_ROLLING_RATING:
        rating = rating_mod.load()
        if rating is not None:
            print(f"  {rating.summary()}")

    master = snapshot(
        players.build(roster, season, teamview, mkts, inputs["starting_lineups"],
                      inputs["fallback_factors"], dc_stats, inputs["dc_params"],
                      improved=improved, factor_history=factor_history, gameweek=gameweek,
                      assist_real=assist_real, rating=rating),
        "players_master")

    if improved:
        rec = reconcile.report(roster, inputs["starting_lineups"], mkts, sportsbet)
        reconcile.print_summary(rec)
        cur = curation_check.check(master)          # availability ceiling + team start-prob sums
        curation_check.print_summary(cur)
        rec = pd.concat([rec, cur], ignore_index=True) if not cur.empty else rec
        if not rec.empty:
            snapshot(rec, "name_reconciliation")

    # Archive updates are destructive (same-gameweek reruns replace rows), so they only
    # happen with an explicit --gw. Inference is a suggestion, never acted on: the FPL
    # data can lag or predate the season, silently corrupting the wrong gameweek.
    if improved:
        if gameweek and archive_mode != "none":
            print(f"Updating historical archives for GW{gameweek}...")
            # Match odds are archived whenever we have them - they cannot be backfilled - unless
            # the W/D/W file is still the synthetic seed (the roll writes one before Betway prices).
            if force_archive or provenance.state(_WDW) != "synthetic":
                history.update_fixture_history(sportsbet["wdw"], season, gameweek=gameweek,
                                               sportsbet=sportsbet, fixtures=inputs["fixtures"])
            else:
                print("  match odds are still the synthetic seed - fixture history not archived")
            if archive_mode == "all":
                withheld = withheld_player_columns(force_archive)
                history.update_player_history(master, gameweek, withhold=withheld)
                history.refresh_fallback_factors(master, withhold=withheld)
        elif gameweek:
            print(f"Archives NOT updated for GW{gameweek} (see guard message above)")
        else:
            guess = history.infer_gameweek()
            hint = f" (FPL data suggests GW{guess})" if guess else ""
            print(f"Archives not updated - rerun with --gw N to record this run{hint}")

    cols = ["Player Name", "Position", "Team", "Cost", "F1 XP"]
    print("\nTop 5 by F1 XP:")
    print(master.nlargest(5, "F1 XP")[cols].to_string(index=False))

    val = master[master["Cost"] > 0].copy()
    val["XP/£"] = (val["F1 XP"] / val["Cost"]).round(3)
    topv = val.nlargest(5, "XP/£")[["Player Name", "Position", "Team", "Cost", "F1 XP", "XP/£"]]
    print("\nTop 5 by value (F1 XP per £):")
    print(topv.to_string(index=False))
    return master


def _parse_gw(argv):
    if "--gw" in argv:
        return int(argv[argv.index("--gw") + 1])
    return None


if __name__ == "__main__":
    parity = "--validate" in sys.argv
    master = run(parity_mode=parity, gameweek=_parse_gw(sys.argv),
                 force_archive="--force-archive" in sys.argv)
    if parity:
        from . import validate
        validate.run(master)
