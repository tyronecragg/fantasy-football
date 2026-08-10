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

from . import config, history, ingest, markets, model, players, reconcile, team_model
from .io_utils import reset_counter, snapshot


def guard_synthetic_archive(gameweek, force=False):
    """Refuse --gw archive recording while the player odds are pre-season placeholders.

    tools/build_preseason_data.py drops sportsbet/SYNTHETIC_NOTE.txt when it writes
    synthetic odds; sportsbet.py removes it after a real player-market scrape. Archiving
    synthetic odds would poison the trailing-median factors, so it needs --force-archive.
    """
    note = os.path.join(config.SPORTSBET_DIR, "SYNTHETIC_NOTE.txt")
    if gameweek and os.path.exists(note) and not force:
        raise SystemExit(
            f"Refusing to record GW{gameweek} archives: {note} says the player odds are "
            f"synthetic pre-season placeholders. Run sportsbet.py for real odds first, "
            f"or pass --force-archive if you really mean it.")


def run(parity_mode=False, gameweek=None, force_archive=False):
    guard_synthetic_archive(gameweek, force_archive)
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
        return _run(parity_mode, gameweek, improved)
    finally:
        config.FPL_DATA_DIR = live_data_dir


def _run(parity_mode, gameweek, improved):
    print("Loading sources...")
    if parity_mode:
        inputs = ingest.load_inputs(config.PARITY_INPUTS_DIR)
        sportsbet = ingest.load_sportsbet(config.PARITY_SPORTSBET_DIR)
    else:
        inputs = ingest.load_inputs()
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

    mkts = markets.build_all(sportsbet, inputs, dedup_f2=improved)
    for key in ("score1", "score2", "assist", "yellow", "clean_sheet", "concede", "gk_saves"):
        snapshot(mkts[key], f"market_{key}")

    # The master must stay snapshot #13 — the optimisers read outputs/13_players_master.csv
    # by name, so variable-count stages (reconciliation) come after it.
    factor_history = None
    if improved:
        factor_history = {stat: h for stat in config.MEDIAN_FACTOR_STATS
                          if (h := history.season_weekly_factors(stat))}
        if factor_history:
            print(f"  trailing-median factors active for {len(factor_history)} stats "
                  f"({len(next(iter(factor_history.values())))} players with archive weeks)")

    master = snapshot(
        players.build(roster, season, teamview, mkts, inputs["starting_lineups"],
                      inputs["fallback_factors"], dc_stats, inputs["dc_params"],
                      improved=improved, factor_history=factor_history),
        "players_master")

    if improved:
        rec = reconcile.report(roster, inputs["starting_lineups"], mkts)
        reconcile.print_summary(rec)
        if not rec.empty:
            snapshot(rec, "name_reconciliation")

    # Archive updates are destructive (same-gameweek reruns replace rows), so they only
    # happen with an explicit --gw. Inference is a suggestion, never acted on: the FPL
    # data can lag or predate the season, silently corrupting the wrong gameweek.
    if improved:
        if gameweek:
            print(f"Updating historical archives for GW{gameweek}...")
            history.update_player_history(master, gameweek)
            history.update_fixture_history(sportsbet["wdw"], season)
            history.refresh_fallback_factors(master)
        else:
            guess = history.infer_gameweek()
            hint = f" (FPL data suggests GW{guess})" if guess else ""
            print(f"Archives not updated - rerun with --gw N to record this run{hint}")

    top = master.nlargest(10, "Total XP")[["Player Name", "Position", "Team", "Cost", "F1 XP", "Total XP"]]
    print("\nTop 10 by Total XP:")
    print(top.to_string(index=False))
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
