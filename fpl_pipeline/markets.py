"""Convert raw bookmaker odds tables into probability tables.

Column selection is positional (matching the workbook's VLOOKUP column indexes) so a
renamed CSV header can't silently shift a lookup.
"""
import numpy as np
import pandas as pd

from . import config


def implied(odds, margin):
    """1/odds/margin. Zero, negative or missing odds -> NaN (Excel #DIV/0!/#N/A)."""
    odds = pd.to_numeric(odds, errors="coerce")
    out = 1.0 / odds / margin
    return out.where(np.isfinite(out))


def two_sided(over, under):
    """Margin removal from an over/under pair: (1/over) / (1/over + 1/under)."""
    over = pd.to_numeric(over, errors="coerce")
    under = pd.to_numeric(under, errors="coerce")
    inv_o, inv_u = 1.0 / over, 1.0 / under
    out = inv_o / (inv_o + inv_u)
    return out.where(np.isfinite(out))


def player_market(df):
    """Goalscorer / 2+ goals / assist CSVs: player_name, match_id, odds_decimal."""
    return pd.DataFrame({
        "player": df.iloc[:, 0],
        "prob": implied(df.iloc[:, 2], config.MARGIN_PLAYER),
    })


def yellow_market(df):
    """Booking CSV: match_name, date, player_name, odds_decimal."""
    return pd.DataFrame({
        "player": df.iloc[:, 2],
        "prob": implied(df.iloc[:, 3], config.MARGIN_PLAYER),
    })


def clean_sheet_market(df):
    """Clean-sheet CSV: match_name, date, team_name, cs_yes, cs_no. Only 'yes' odds used."""
    return pd.DataFrame({
        "team": df.iloc[:, 2],
        "prob": implied(df.iloc[:, 3], config.MARGIN_PLAYER),
    })


def concede_market(df):
    """Team-goals CSV keyed by *opponent*: the workbook looks up a player's team in the
    Opponent column, so 'prob2/prob4' are P(team concedes 2+/4+) = the row team's goals."""
    p4 = two_sided(df.iloc[:, 6], df.iloc[:, 7])
    return pd.DataFrame({
        "team": df.iloc[:, 3],                              # Opponent column
        "prob2": two_sided(df.iloc[:, 4], df.iloc[:, 5]),   # no IFERROR in the sheet
        "prob4": p4.fillna(0.0),                            # IFERROR(...,0)
    })


def gk_saves_market(df):
    """GK saves CSV: Match, Date, Team, Goalkeeper, 3+ odds, 6+ odds — with defaults."""
    return pd.DataFrame({
        "team": df.iloc[:, 2],
        "prob3": implied(df.iloc[:, 4], config.MARGIN_PLAYER).fillna(config.SAVES3_DEFAULT),
        "prob6": implied(df.iloc[:, 5], config.MARGIN_PLAYER).fillna(config.SAVES6_DEFAULT),
    })


def f2_yellow_market(df):
    """inputs/f2_yellow_card.csv: match_name, date, player_name, odds_decimal.
    Source pipeline was deleted Dec 2025 — usually empty, and the model fallback applies."""
    if df.empty:
        return pd.DataFrame(columns=["player", "prob"])
    return pd.DataFrame({
        "player": df.iloc[:, 2],
        "prob": implied(df.iloc[:, 3], config.MARGIN_PLAYER),
    })


def _without_f1_duplicate(f1_raw, f2_raw, label):
    """A single-gameweek scrape used to make sportsbet.py write the F1 odds into the F2
    files verbatim (tail(20) of a 20-row frame). Wrong-fixture odds are worse than no
    odds, so an F2 file identical to its F1 file is treated as absent."""
    if len(f2_raw) and f2_raw.equals(f1_raw):
        print(f"  note: F2 {label} odds duplicate F1 (single-gameweek scrape) - ignoring them")
        return f2_raw.iloc[0:0]
    return f2_raw


def build_all(sportsbet, inputs, dedup_f2=True):
    f2_cs, f2_tg = sportsbet["f2_clean_sheet"], sportsbet["f2_team_goals"]
    if dedup_f2:
        f2_cs = _without_f1_duplicate(sportsbet["clean_sheet"], f2_cs, "clean-sheet")
        f2_tg = _without_f1_duplicate(sportsbet["team_goals"], f2_tg, "team-goals")
    return {
        "score1": player_market(sportsbet["score1"]),
        "score2": player_market(sportsbet["score2"]),
        "assist": player_market(sportsbet["assist"]),
        "yellow": yellow_market(sportsbet["yellow"]),
        "clean_sheet": clean_sheet_market(sportsbet["clean_sheet"]),
        "concede": concede_market(sportsbet["team_goals"]),
        "gk_saves": gk_saves_market(sportsbet["gk_saves"]),
        "f2_clean_sheet": clean_sheet_market(f2_cs),
        "f2_concede": concede_market(f2_tg),
        "f2_yellow": f2_yellow_market(inputs["f2_yellow"]),
    }
