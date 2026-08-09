"""Load every raw data source into DataFrames.

Ports the roster/DC logic of extract_fpl_data.py and extract_defensive_contributions.py
so the workbook's FPL Players / Defensive Contribution sheets are no longer needed.
"""
import os

import pandas as pd

from . import config
from .io_utils import read_csv_tolerant
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


def _prior_season(season):
    a, b = season.split("-")
    return f"{int(a) - 1}-{int(b) - 1}"


def _dc_source(data_dir):
    players = pd.read_csv(os.path.join(data_dir, "players.csv"))
    player_stats = pd.read_csv(os.path.join(data_dir, "playerstats.csv"))
    player_stats = player_stats.sort_values("gw").drop_duplicates(subset="id", keep="last")
    df = players.merge(player_stats[["id", "minutes", "defensive_contribution_per_90"]],
                       left_on="player_id", right_on="id", how="left")
    df["name"] = _apply_name_changes(df["first_name"] + " " + df["second_name"])
    df["nineties"] = (df["minutes"] / 90).round(2)
    df["position"] = df["position"].map(config.POSITION_MAP)
    return df.rename(columns={"defensive_contribution_per_90": "dc90"})


def load_defensive_contributions():
    """Per-position DC stats: name, 90s, dc_per_90.

    Current and prior-season DC-per-90 are blended, weighted by minutes played, with
    the prior season's weight capped at config.DC_PRIOR_CAP_MINUTES: pure prior at the
    season start (current minutes ~0), fading as the current season accumulates.
    Name-keyed, so club changes carry a player's DC identity. The blended minutes also
    drive the >=4-nineties reliability gate; players with no data in either season
    still land on the position average."""
    df = _dc_source(config.FPL_DATA_DIR)

    m_cur = df["minutes"].fillna(0.0).where(df["dc90"].notna(), 0.0)
    num = m_cur * df["dc90"].fillna(0.0)

    prior_dir = os.path.join(os.path.dirname(config.FPL_DATA_DIR), _prior_season(config.SEASON))
    if os.path.isdir(prior_dir):
        prior = (_dc_source(prior_dir).dropna(subset=["dc90"])
                 .drop_duplicates(subset="name").set_index("name"))
        m_pri = (df["name"].map(prior["minutes"]).fillna(0.0)
                 .clip(upper=config.DC_PRIOR_CAP_MINUTES))
        num = num + m_pri * df["name"].map(prior["dc90"]).fillna(0.0)
        weight = m_cur + m_pri
    else:
        weight = m_cur

    df["dc90"] = (num / weight).where(weight > 0)
    df["nineties"] = (weight / 90).round(2)

    out = {}
    for pos in ("DEF", "MID"):
        out[pos] = (
            df[df["position"] == pos][["name", "nineties", "dc90"]]
            .sort_values("name")
            .reset_index(drop=True)
        )
    return out


def _workbook_values(sheet_name, n_cols):
    """Values-only read of a workbook sheet's first n_cols columns (parity source)."""
    import openpyxl

    wb = openpyxl.load_workbook(config.WORKBOOK, data_only=True, read_only=True)
    ws = wb[sheet_name]
    rows = [row[:n_cols] for row in ws.values]
    wb.close()
    df = pd.DataFrame(rows[1:], columns=range(n_cols))
    return df.dropna(how="all")


def load_fpl_players_workbook():
    """Roster as frozen in the workbook's FPL Players sheet. Used by parity mode (and
    tests) so they compare against the workbook's own inputs — the upstream
    FPL-Core-Insights repo rewrites past-season data and would break alignment."""
    df = _workbook_values("FPL Players", 4)
    df.columns = ["name", "position", "team", "cost"]
    return df.reset_index(drop=True)


def load_defensive_contributions_workbook():
    """DC stats as frozen in the workbook's Defensive Contribution sheets (cols A-C)."""
    out = {}
    for pos, sheet in [("DEF", "Defensive Contribution DEF"), ("MID", "Defensive Contribution MID")]:
        df = _workbook_values(sheet, 3)
        df.columns = ["name", "nineties", "dc90"]
        out[pos] = df.reset_index(drop=True)
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


def load_sportsbet(base_dir=None):
    base_dir = base_dir or config.SPORTSBET_DIR
    return {key: read_csv_tolerant(os.path.join(base_dir, fname))
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


def load_inputs(base_dir=None):
    base_dir = base_dir or config.INPUTS_DIR
    return {key: read_csv_tolerant(os.path.join(base_dir, fname))
            for key, fname in INPUT_FILES.items()}
