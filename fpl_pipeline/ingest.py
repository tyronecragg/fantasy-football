"""Load every raw data source into DataFrames.

Ports the roster/DC logic of extract_fpl_data.py and extract_defensive_contributions.py
so the workbook's FPL Players / Defensive Contribution sheets are no longer needed.
"""
import os

import pandas as pd

from . import config
from .names import apply_player_names as _apply_name_changes


def load_fpl_players():
    """Roster: name, position, team, cost — identical logic to extract_fpl_data.py."""
    players = pd.read_csv(os.path.join(config.FPL_DATA_DIR, "players.csv"))
    player_stats = pd.read_csv(os.path.join(config.FPL_DATA_DIR, "playerstats.csv"))
    teams = pd.read_csv(os.path.join(config.FPL_DATA_DIR, "teams.csv"))

    team_map = teams.set_index("code")["name"].to_dict()
    players["team"] = players["team_code"].map(team_map)
    players["position"] = players["position"].map(config.POSITION_MAP)

    player_stats = player_stats.sort_values("gw").drop_duplicates(subset="id", keep="last")
    df = players.merge(player_stats[["id", "now_cost"]], left_on="player_id", right_on="id", how="left")

    df["name"] = _apply_name_changes(df["first_name"] + " " + df["second_name"])
    df = df.rename(columns={"now_cost": "cost"})
    df["position"] = df["position"].astype(pd.CategoricalDtype(categories=config.POSITION_ORDER, ordered=True))

    return (
        df[["name", "position", "team", "cost"]]
        .sort_values(["team", "position", "name", "cost"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )


def load_defensive_contributions():
    """Per-position DC stats: name, 90s, dc_per_90 — as extract_defensive_contributions.py."""
    players = pd.read_csv(os.path.join(config.FPL_DATA_DIR, "players.csv"))
    player_stats = pd.read_csv(os.path.join(config.FPL_DATA_DIR, "playerstats.csv"))

    players["position"] = players["position"].map(config.POSITION_MAP)
    player_stats = player_stats.sort_values("gw").drop_duplicates(subset="id", keep="last")
    df = players.merge(player_stats[["id", "minutes", "defensive_contribution_per_90"]],
                       left_on="player_id", right_on="id", how="left")
    df["name"] = _apply_name_changes(df["first_name"] + " " + df["second_name"])
    df["nineties"] = (df["minutes"] / 90).round(2)
    df = df.rename(columns={"defensive_contribution_per_90": "dc90"})

    out = {}
    for pos in ("DEF", "MID"):
        out[pos] = (
            df[df["position"] == pos][["name", "nineties", "dc90"]]
            .sort_values("name")
            .reset_index(drop=True)
        )
    return out


SPORTSBET_FILES = {
    "wdw": "sportsbet_win_draw_win_odds.csv",
    "score1": "sportsbet_goalscorer_odds.csv",
    "score2": "sportsbet_two_goals_odds.csv",
    "assist": "sportsbet_assist_odds.csv",
    "yellow": "sportsbet_booking_odds.csv",
    "clean_sheet": "sportsbet_clean_sheet_odds.csv",
    "team_goals": "sportsbet_team_goals_odds.csv",
    "gk_saves": "sportsbet_goalkeeper_saves_odds.csv",
    "f2_clean_sheet": "sportsbet_clean_sheet_odds_f2.csv",
    "f2_team_goals": "sportsbet_team_goals_odds_f2.csv",
}


def load_sportsbet():
    return {key: pd.read_csv(os.path.join(config.SPORTSBET_DIR, fname))
            for key, fname in SPORTSBET_FILES.items()}


INPUT_FILES = {
    "title_odds": "title_odds.csv",
    "relegation_odds": "relegation_odds.csv",
    "top6_odds": "top6_odds.csv",
    "fixtures": "fixtures.csv",
    "starting_lineups": "starting_lineups.csv",
    "gw_teams": "gw_teams.csv",
    "fallback_factors": "fallback_factors.csv",
    "f2_yellow": "f2_yellow_card.csv",
    "dc_params": "dc_params.csv",
}


def load_inputs():
    return {key: pd.read_csv(os.path.join(config.INPUTS_DIR, fname))
            for key, fname in INPUT_FILES.items()}
