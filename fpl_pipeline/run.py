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

from . import history, ingest, markets, model, players, team_model
from .io_utils import reset_counter, snapshot


def run(parity_mode=False, gameweek=None):
    reset_counter()
    improved = not parity_mode

    if parity_mode:
        print("Running in parity mode (workbook-exact, improvements disabled)")
        if os.path.exists(model.WORKBOOK_COEFFICIENTS_JSON):
            print("  using workbook-extracted coefficients (coefficients_workbook.json)")
            model.load_coefficients(model.WORKBOOK_COEFFICIENTS_JSON)
    else:
        model.load_coefficients()

    print("Loading sources...")
    inputs = ingest.load_inputs()
    sportsbet = ingest.load_sportsbet()

    roster = snapshot(ingest.load_fpl_players(), "fpl_players")
    dc_stats = ingest.load_defensive_contributions()
    snapshot(dc_stats["DEF"], "dc_stats_def")
    snapshot(dc_stats["MID"], "dc_stats_mid")

    season = snapshot(team_model.season_probs(inputs), "season_probs")
    teamview = snapshot(team_model.team_fixture_view(inputs, sportsbet, draw_aware=improved),
                        "team_fixture_view")

    mkts = markets.build_all(sportsbet, inputs, dedup_f2=improved)
    for key in ("score1", "score2", "assist", "yellow", "clean_sheet", "concede", "gk_saves"):
        snapshot(mkts[key], f"market_{key}")

    master = snapshot(
        players.build(roster, season, teamview, mkts, inputs["starting_lineups"],
                      inputs["fallback_factors"], dc_stats, inputs["dc_params"],
                      improved=improved),
        "players_master")

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
    master = run(parity_mode=parity, gameweek=_parse_gw(sys.argv))
    if parity:
        from . import validate
        validate.run(master)
