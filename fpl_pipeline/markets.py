"""Convert raw bookmaker odds tables into probability tables.

Column selection is positional (matching the workbook's VLOOKUP column indexes) so a
renamed CSV header can't silently shift a lookup.
"""
import numpy as np
import pandas as pd

from . import config, names


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


def longshot_calibrate(prob, enabled=True):
    """Shrink over-stated long prices toward their observed frequency.

    MARGIN_PLAYER removes a flat 5% at every price. Measured against outcomes that is about
    right above ~18% and far too little below it — a 5-8% projection happened 2.8% of the
    time. Applied here, at the point odds become probabilities, because that is the stage
    the calibration was measured on; the scraped CSVs stay raw so a different curve can be
    fitted later.

    Interpolates in LOG probability because the distortion grows multiplicatively as the
    price lengthens, and floors the multiplier so no player is declared impossible.
    """
    knots = getattr(config, "LONGSHOT_CALIBRATION", None)
    if not enabled or not knots:
        return prob
    xs = np.log([k[0] for k in knots])
    ys = np.maximum([k[1] for k in knots], getattr(config, "LONGSHOT_FLOOR", 0.15))
    p = pd.to_numeric(prob, errors="coerce")
    mult = np.interp(np.log(p.clip(lower=1e-6)), xs, ys)
    return (p * mult).clip(0, 1).where(p.notna())


def player_market(df, calibrate=False, roster_names=None):
    """Goalscorer / 2+ goals / assist CSVs: player_name, match_id, odds_decimal.

    `calibrate` is improved-mode only. Parity mode must reproduce the workbook exactly, and
    the workbook applied a flat margin, so the longshot correction is a deliberate
    divergence and stays off there.

    Improved mode also resolves names at READ time (the roster-side name_mappings, then an
    accent/case-insensitive match to the roster), mirroring yellow_market. Betway cleans names
    on write too, but that only helps future scrapes — read-time resolution means a mapping
    added after a scrape takes effect on the next plain pipeline run, no re-scrape needed.
    """
    players = df.iloc[:, 0]
    if calibrate:
        players = names.apply_player_names(players)
        if roster_names is not None:
            players = names.resolve_to_roster(players, roster_names)
    return pd.DataFrame({
        "player": players,
        "prob": longshot_calibrate(implied(df.iloc[:, 2], config.MARGIN_PLAYER), calibrate),
    })


def yellow_market(df, roster_names=None, improved=False):
    """Booking CSV: match_name, date, player_name, odds_decimal.

    Bet365-sourced and collected by hand, so names are resolved independently of the
    roster-side name_mappings the Betway markets rely on: the Bet365-only mapping first
    (genuine spelling/form differences), then an accent/case-insensitive match to the roster,
    so a stray accent can't silently drop a card. Improved mode only — parity keeps the exact
    passthrough and the MARGIN_PLAYER divisor so its output stays byte-identical. The card
    market runs a heavier overround than goals/assists, hence MARGIN_CARD."""
    names_col = df.iloc[:, 2]
    margin = config.MARGIN_PLAYER
    if improved:
        margin = config.MARGIN_CARD
        names_col = names.apply_bet365_names(names_col)
        if roster_names is not None:
            names_col = names.resolve_to_roster(names_col, roster_names)
    return pd.DataFrame({
        "player": names_col,
        "prob": implied(df.iloc[:, 3], margin),
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
    """A single-gameweek scrape used to make the old Sportsbet scraper write the F1 odds into the F2
    files verbatim (tail(20) of a 20-row frame). Wrong-fixture odds are worse than no
    odds, so an F2 file identical to its F1 file is treated as absent."""
    if len(f2_raw) and f2_raw.equals(f1_raw):
        print(f"  note: F2 {label} odds duplicate F1 (single-gameweek scrape) - ignoring them")
        return f2_raw.iloc[0:0]
    return f2_raw


def build_all(sportsbet, inputs, dedup_f2=True, calibrate=None, roster_names=None):
    # calibration rides with improved mode unless told otherwise
    calibrate = dedup_f2 if calibrate is None else calibrate
    f2_cs, f2_tg = sportsbet["f2_clean_sheet"], sportsbet["f2_team_goals"]
    if dedup_f2:
        f2_cs = _without_f1_duplicate(sportsbet["clean_sheet"], f2_cs, "clean-sheet")
        f2_tg = _without_f1_duplicate(sportsbet["team_goals"], f2_tg, "team-goals")
    return {
        "score1": player_market(sportsbet["score1"], calibrate, roster_names),
        "score2": player_market(sportsbet["score2"], calibrate, roster_names),
        "assist": player_market(sportsbet["assist"], calibrate, roster_names),
        "assist2": player_market(sportsbet["assist2"], calibrate, roster_names),
        "yellow": yellow_market(sportsbet["yellow"], roster_names, improved=calibrate),
        "clean_sheet": clean_sheet_market(sportsbet["clean_sheet"]),
        "concede": concede_market(sportsbet["team_goals"]),
        "gk_saves": gk_saves_market(sportsbet["gk_saves"]),
        "f2_clean_sheet": clean_sheet_market(f2_cs),
        "f2_concede": concede_market(f2_tg),
        "f2_yellow": f2_yellow_market(inputs["f2_yellow"]),
    }
