"""Load every raw data source into DataFrames.

Ports the roster/DC logic of extract_fpl_data.py and extract_defensive_contributions.py
so the workbook's FPL Players / Defensive Contribution sheets are no longer needed.
"""
import os

import pandas as pd

from . import config
from .io_utils import read_csv_tolerant
from .names import apply_player_names as _apply_name_changes, apply_team_names as _apply_team_names


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
        df[["player_id", "name", "position", "team", "cost"]]
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
    teams = pd.read_csv(os.path.join(data_dir, "teams.csv")).set_index("code")["name"]
    df["team"] = _apply_team_names(df["team_code"].map(teams))     # normalised, to detect club changes
    return df.rename(columns={"defensive_contribution_per_90": "dc90"})


def _merge_external_prior(prior):
    """Add face-value DefCon priors (Championship / foreign leagues, from tools/build_dc_prior.py)
    for players the FPL PL prior has no row for. FPL history always wins on overlap; the external
    rows only FILL GAPS. Columns aligned to the FPL prior: name-indexed dc90, minutes, team. Same-club
    promoted players carry their own club as the team, so the mover discount does NOT fire (face value)."""
    path = config.EXTERNAL_DC_PRIOR
    if not os.path.exists(path):
        return prior
    ext = pd.read_csv(path).dropna(subset=["dc90"]).drop_duplicates(subset="name").set_index("name")
    new = ext[~ext.index.isin(prior.index)][["minutes", "team", "dc90"]]
    return pd.concat([prior[["minutes", "team", "dc90"]], new]) if len(new) else prior


def load_defensive_contributions():
    """Per-position DC stats: name, 90s, dc_per_90.

    Current and prior-season DC-per-90 are blended, weighted by minutes played, with
    the prior season's weight capped at config.DC_PRIOR_CAP_MINUTES: pure prior at the
    season start (current minutes ~0), fading as the current season accumulates. Current
    minutes are weighted config.DC_CURRENT_SEASON_WEIGHT times prior ones so recent form
    counts for more. Name-keyed, so club changes carry a player's DC identity.

    TWO denominators, deliberately: the recency weight tilts the RATE (dc90) but NOT the
    EVIDENCE count. `nineties` is TRUE minutes / 90 (current + capped prior, undoubled) and
    drives the >=4-nineties reliability gate and the shrinkage in players._dc_table - a
    recency preference is not extra evidence, so one recent match stays one match of trust.
    Players with no data in either season still land on the position average."""
    df = _dc_source(config.FPL_DATA_DIR)

    w = config.DC_CURRENT_SEASON_WEIGHT
    m_cur = df["minutes"].fillna(0.0).where(df["dc90"].notna(), 0.0)
    num = w * m_cur * df["dc90"].fillna(0.0)

    prior_dir = os.path.join(os.path.dirname(config.FPL_DATA_DIR), _prior_season(config.SEASON))
    if os.path.isdir(prior_dir):
        prior = (_dc_source(prior_dir).dropna(subset=["dc90"])
                 .drop_duplicates(subset="name").set_index("name"))
        prior = _merge_external_prior(prior)
        m_pri = (df["name"].map(prior["minutes"]).fillna(0.0)
                 .clip(upper=config.DC_PRIOR_CAP_MINUTES))
        # MOVERS: a player at a different club than last season has a stale role, so his prior counts
        # only DC_MOVER_PRIOR_WEIGHT in the RATE blend (not the evidence). Requires both teams known.
        prior_team = df["name"].map(prior["team"])
        mover = df["team"].notna() & prior_team.notna() & (df["team"] != prior_team)
        prior_rate = m_pri * mover.map({True: config.DC_MOVER_PRIOR_WEIGHT, False: 1.0}).astype(float)
        num = num + prior_rate * df["name"].map(prior["dc90"]).fillna(0.0)
        rate_denom = w * m_cur + prior_rate   # recency- and mover-tilted, for the dc90 weighted average
        evidence = m_cur + m_pri              # true minutes (mover discount does NOT touch evidence)
    else:
        rate_denom = w * m_cur
        evidence = m_cur

    df["dc90"] = (num / rate_denom).where(rate_denom > 0)
    df["nineties"] = (evidence / 90).round(2)

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
    "assist2": "sportsbet_two_assists_odds.csv",
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


# Optional inputs: absent file -> empty frame with these columns (no error)
OPTIONAL_INPUT_FILES = {
    "season_odds_corrections": ("season_odds_corrections.csv",
                                ["market", "Team", "corrected_odds", "reason"]),
}

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
    out = {key: read_csv_tolerant(os.path.join(base_dir, fname))
           for key, fname in INPUT_FILES.items()}
    for key, (fname, columns) in OPTIONAL_INPUT_FILES.items():
        path = os.path.join(base_dir, fname)
        out[key] = read_csv_tolerant(path) if os.path.exists(path) else pd.DataFrame(columns=columns)
    return out
