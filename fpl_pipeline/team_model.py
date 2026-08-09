"""Team-level model: season probabilities (Overall Odds) and the per-team fixture view
(Team Fixture Odds), replicated from the workbook formulas including their quirks.
"""
import numpy as np
import pandas as pd

from . import config
from .markets import implied
from .names import apply_team_names


def season_probs(inputs, workbook_quirks=False):
    """Replicates the Overall Odds sheet: average each row's bookmaker odds, with 0
    mapped to a 'no market' sentinel (5001 title/top6, 2001 relegation).

    workbook_quirks=True (parity mode only) additionally applies two 2025-26-specific
    hacks baked into the sheet: Man City's relegation odds taken from the team in sheet
    row 6 ($B$6), and Wolverhampton's 0 relegation odds meaning 'already relegated' -> 1.
    """
    title, releg, top6 = inputs["title_odds"], inputs["relegation_odds"], inputs["top6_odds"]

    def averages(df):
        return pd.Series(df.iloc[:, 1:].mean(axis=1, skipna=True).values, index=df["Team"])

    title_avg, releg_avg, top6_avg = averages(title), averages(releg), averages(top6)

    teams_raw = title["Team"]                     # Overall Odds row order = Title Odds order

    title_odds = title_avg.reindex(teams_raw)
    title_odds = title_odds.mask(title_odds == 0, config.SENTINEL_TITLE_TOP6)

    r = releg_avg.reindex(teams_raw)
    if workbook_quirks:
        row6_team = teams_raw.iloc[4]             # $B$6: 5th data row
        releg_odds = r.mask(r == 0, np.where(teams_raw == "Wolverhampton", 1.0,
                                             config.SENTINEL_RELEGATION))
        releg_odds = releg_odds.mask((teams_raw == "Man City").values,
                                     releg_avg.get(row6_team, np.nan))
    else:
        releg_odds = r.mask(r == 0, config.SENTINEL_RELEGATION)

    top6_odds = top6_avg.reindex(teams_raw)
    top6_odds = top6_odds.mask(top6_odds == 0, config.SENTINEL_TITLE_TOP6)

    out = pd.DataFrame({
        "team": apply_team_names(teams_raw).values,
        "team_raw": teams_raw.values,
        "title_odds": title_odds.values,
        "relegation_odds": releg_odds.values,
        "top6_odds": top6_odds.values,
    })
    for col in ("title", "relegation", "top6"):
        out[col] = implied(out[f"{col}_odds" if col != "top6" else "top6_odds"], config.MARGIN_SEASON)
    return out


def _win_prob_lookup(teams, wdw_block, draw_aware=False):
    """Win probability for a team within one gameweek's WDW block; NaN when the team
    has no match in the block.

    Legacy (workbook) mode: 1/odds/1.03 per side independently.
    Draw-aware mode (improved, needs a draw_odds column from the scraper): proper
    three-way de-margining, p_side = (1/side) / (1/home + 1/draw + 1/away). Rows
    without draw odds fall back to the legacy formula.
    """
    block = wdw_block.reset_index(drop=True)
    h = pd.to_numeric(block.iloc[:, 2], errors="coerce")
    a = pd.to_numeric(block.iloc[:, 3], errors="coerce")

    if draw_aware and "draw_odds" in block.columns:
        d = pd.to_numeric(block["draw_odds"], errors="coerce")
        overround = 1 / h + 1 / d + 1 / a
        p_home = (1 / h / overround).where(d.notna(), implied(h, config.MARGIN_WDW))
        p_away = (1 / a / overround).where(d.notna(), implied(a, config.MARGIN_WDW))
    else:
        p_home = implied(h, config.MARGIN_WDW)
        p_away = implied(a, config.MARGIN_WDW)

    home_probs = pd.Series(p_home.values, index=block.iloc[:, 0]).groupby(level=0).first()
    away_probs = pd.Series(p_away.values, index=block.iloc[:, 1]).groupby(level=0).first()
    return teams.map(home_probs).where(teams.isin(home_probs.index), teams.map(away_probs))


def team_fixture_view(inputs, sportsbet, draw_aware=False):
    """Replicates the Team Fixture Odds sheet: one row per team with F1/F2 win odds and
    F1–F8 opponent/venue from the Fixtures input (missing gameweeks -> NaN)."""
    fixtures = inputs["fixtures"]
    wdw = sportsbet["wdw"]
    teams = fixtures.iloc[:, 0]

    def fixture_col(idx):
        return fixtures.iloc[:, idx] if idx < fixtures.shape[1] else pd.Series(np.nan, index=fixtures.index)

    out = pd.DataFrame({"team": teams})
    out["f1_opponent"] = fixture_col(1).values
    out["f1_venue"] = fixture_col(2).values
    out["f2_opponent"] = fixture_col(3).values
    out["f2_venue"] = fixture_col(4).values
    for k in range(3, 9):
        out[f"f{k}_opponent"] = fixture_col(5 + (k - 3) * 2).values
        out[f"f{k}_venue"] = fixture_col(6 + (k - 3) * 2).values

    f1_block, f2_block = wdw.iloc[:10], wdw.iloc[10:20]
    out["f1_win"] = _win_prob_lookup(out["team"], f1_block, draw_aware).values
    out["f1_opponent_win"] = _win_prob_lookup(pd.Series(out["f1_opponent"]), f1_block, draw_aware).values
    if len(f2_block):
        out["f2_win"] = _win_prob_lookup(out["team"], f2_block, draw_aware).values
        out["f2_opponent_win"] = _win_prob_lookup(pd.Series(out["f2_opponent"]), f2_block, draw_aware).values
    else:
        out["f2_win"] = np.nan
        out["f2_opponent_win"] = np.nan
    return out
