# -*- coding: utf-8 -*-
"""In-season synthetic F1 markets for a new gameweek, from a PAST gameweek's ACTUAL factors
re-cast onto the new fixtures — the mid-season sibling of build_preseason_data.

    env/Scripts/python tools/build_synthetic_gw.py --gw N [--factor-gw 1] [--factor-season S]

Rolling to GW N: run `build_fixtures --gw N` and update the outright odds FIRST, then this. For
each player it takes their `--factor-gw` (default 1) actual market factor from the archive
(factor = archived P(stat) / baseline at that fixture's win probs) and applies it to the NEW
fixture's baseline (P = factor x baseline at GW N's win probs, from win_pred on the updated odds).
Team markets come from the win probs. ONLY F1 (GW N) is seeded — synthetic odds apply to F1 alone;
F2+ are derived off the F1 synthetic data by the pipeline (model / factor x baseline). Odds carry the
usual margins so the pipeline's de-margining recovers the intended probabilities, and it drops
the provenance manifest (fpl_pipeline/provenance.py) so `--gw` archiving withholds player history until real odds land.

This OVERWRITES the sportsbet/*.csv F1 files. Running tools/betway.py afterwards UPSERTS each
market Betway-authoritatively PER MATCH: real odds replace synthetic for every match Betway prices
(a player Betway omits from a priced match becomes NA — it didn't rate him), while wholly unpriced
matches keep this synthetic data.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fpl_pipeline import config, ingest, model, provenance, team_model  # noqa: E402
from fpl_pipeline.players import normalize_start_probs  # noqa: E402
from tools.build_preseason_data import (  # noqa: E402  (reuse the tested helpers)
    FACTOR_STATS, PLAYER_MARGIN, WDW_MARGIN, fixture_win_probs)

HIST = os.path.join(config.INPUTS_DIR, "historical_player_data.csv")
EPOCH = 1788030000  # ~28 Aug 2026, informational (the pipeline keys markets by player/team, not date)


def gw_factors(season, gw):
    """Each player's ACTUAL market factor from one archived gameweek: factor = archived P(stat)
    / baseline(stat) at that gameweek's own win probs. One row per player (median if repeated)."""
    hist = pd.read_csv(HIST, low_memory=False)
    for c in hist.columns:
        if c not in ("Season", "Player Name", "Position", "Team", "F1 Opponent", "F1 Venue"):
            hist[c] = pd.to_numeric(hist[c], errors="coerce")
    h = hist[(hist["Season"].astype(str) == str(season)) & (hist["Gameweek"] == gw)].copy()
    if h.empty:
        raise SystemExit(f"no archived rows for {season} GW{gw} — cannot source factors")
    win, opp, pos, home = h["F1 Win"], h["F1 Opponent Win"], h["Position"], h["F1 Venue"] == "H"
    out = pd.DataFrame({"Player Name": h["Player Name"]})
    for col, (stat, prob_col) in FACTOR_STATS.items():
        out[col] = h[prob_col] / model.baseline(stat, win, opp, pos, home)
    print(f"  sourced GW{gw} {season} factors for {h['Player Name'].nunique()} players")
    return out.groupby("Player Name", sort=True).median().reset_index()


def _to_odds(p, margin=PLAYER_MARGIN):
    return (1 / (p.clip(0.01, 0.95) * margin)).round(2)


def _team_side_view(gw):
    """Long per-team frame from a per-match win-prob frame: Team, opp, venue, p (own win), po (opp)."""
    return pd.concat([
        gw.rename(columns={"home_team": "Team", "away_team": "opp"}).assign(
            venue="H", p=lambda d: d.p_home, po=lambda d: d.p_away),
        gw.rename(columns={"away_team": "Team", "home_team": "opp"}).assign(
            venue="A", p=lambda d: d.p_away, po=lambda d: d.p_home),
    ], ignore_index=True)[["Team", "opp", "venue", "p", "po"]]


def _team_block(gw, epoch):
    """Clean-sheet and team-goals blocks from win probs (mirrors build_preseason_data.team_block)."""
    rows = _team_side_view(gw)
    anypos = pd.Series("MID", index=rows.index)
    home = rows["venue"] == "H"
    p_cs = model.baseline("clean_sheet", rows["p"], rows["po"], anypos, home).clip(0.03, 0.7)
    p2 = model.baseline("concede2", rows["po"], rows["p"], anypos, ~home).clip(0.05, 0.9)
    p4 = model.baseline("concede4", rows["po"], rows["p"], anypos, ~home).clip(0.01, 0.6)
    cs = pd.DataFrame({"match_name": rows["Team"] + " v " + rows["opp"], "date": epoch,
                       "team_name": rows["Team"],
                       "clean_sheet_yes": _to_odds(p_cs), "clean_sheet_no": _to_odds(1 - p_cs)})
    tg = pd.DataFrame({"Match": rows["Team"] + " v " + rows["opp"], "Date": epoch,
                       "Team": rows["Team"], "Opponent": rows["opp"],
                       "Team_Over_1.5": _to_odds(p2), "Team_Under_1.5": _to_odds(1 - p2),
                       "Team_Over_3.5": _to_odds(p4), "Team_Under_3.5": _to_odds(1 - p4)})
    for c in ("Over_1.5", "Under_1.5", "Over_3.5", "Under_3.5"):
        tg[f"Opponent_Concedes_{c}"] = tg[f"Team_{c}"]
    return cs, tg


def _wdw_rows(frame):
    out = frame.copy()
    out["home_win_odds"] = (1 / (out["p_home"] * WDW_MARGIN)).round(2)
    out["away_win_odds"] = (1 / (out["p_away"] * WDW_MARGIN)).round(2)
    out["draw_odds"] = (1 / ((1 - out["p_home"] - out["p_away"]).clip(0.10) * WDW_MARGIN)).round(2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, required=True, help="the new F1 gameweek (F2 = gw+1)")
    ap.add_argument("--factor-gw", type=int, default=None,
                    help="archived gameweek to source player factors from (default: gw-1, the last played week)")
    ap.add_argument("--factor-season", default=config.SEASON)
    a = ap.parse_args()
    if a.factor_gw is None:
        a.factor_gw = a.gw - 1
    sb = config.SPORTSBET_DIR

    roster = ingest.load_fpl_players()
    season = team_model.season_probs(ingest.load_inputs())      # uses the UPDATED outright odds
    fb = gw_factors(a.factor_season, a.factor_gw).set_index("Player Name")
    med = fb.median(numeric_only=True)                          # for players with no factor-gw row
    lineups = normalize_start_probs(pd.read_csv(os.path.join(config.INPUTS_DIR, "starting_lineups.csv")))
    sf = pd.read_csv(os.path.join(config.INPUTS_DIR, "season_fixtures.csv"))
    gw1 = fixture_win_probs(season, sf[sf["gameweek"] == a.gw])        # F1 = new gameweek
    if gw1.empty:
        raise SystemExit(f"no fixtures for GW{a.gw} in season_fixtures.csv")

    # --- win/draw/win: F1 ONLY. Synthetic seeds F1 markets exclusively; F2+ are DERIVED off the F1
    # synthetic data (model / factor x baseline), so no synthetic F2 block is written — a NaN F2 in the
    # market files makes the pipeline project F2. Real Betway F2 odds still repopulate/override.
    cols = ["home_team", "away_team", "home_win_odds", "away_win_odds", "draw_odds"]
    _wdw_rows(gw1)[cols].to_csv(os.path.join(sb, "sportsbet_win_draw_win_odds.csv"), index=False)

    # --- per-player context on the new F1 fixture (each team's own win/opp/venue)
    ctx = lineups.merge(_team_side_view(gw1), on="Team", how="left")
    ctx["position"] = ctx["Player"].map(ingest.load_fpl_players().set_index("name")["position"]).fillna("MID").astype(str)
    win, opp, home, pos = ctx["p"], ctx["po"], ctx["venue"] == "H", ctx["position"]
    match_name = ctx["Team"] + " v " + ctx["opp"]
    mid = 90000000 + ctx.index
    outfield = pos != "GK"

    def fac(col):
        return ctx["Player"].map(fb[col]).fillna(med[col])

    p_score = (fac("Score 1+ Factor") * model.baseline("score1", win, opp, pos, home)).clip(0.01, 0.9)
    p_assist = (fac("Assist Factor") * model.baseline("assist", win, opp, pos, home)).clip(0.01, 0.9)
    p_yellow = fac("F1 Yellow Card Factor") * model.baseline("yellow", win, opp, pos, home)

    pd.DataFrame({"player_name": ctx["Player"], "match_id": mid,
                  "odds_decimal": _to_odds(p_score)})[outfield].to_csv(
        os.path.join(sb, "sportsbet_goalscorer_odds.csv"), index=False)
    pd.DataFrame({"player_name": ctx["Player"], "match_id": mid,
                  "odds_decimal": _to_odds(model.poisson_score2(p_score))})[outfield].to_csv(
        os.path.join(sb, "sportsbet_two_goals_odds.csv"), index=False)
    pd.DataFrame({"player_name": ctx["Player"], "match_id": mid,
                  "odds_decimal": _to_odds(p_assist)})[outfield].to_csv(
        os.path.join(sb, "sportsbet_assist_odds.csv"), index=False)
    pd.DataFrame({"match_name": match_name, "date": EPOCH, "player_name": ctx["Player"],
                  "odds_decimal": _to_odds(p_yellow)})[outfield].to_csv(
        os.path.join(sb, "sportsbet_booking_odds.csv"), index=False)

    gk = ctx[pos == "GK"]
    ghome = gk["venue"] == "H"
    p3 = (gk["Player"].map(fb["F1 3+ Saves Factor"]).fillna(med["F1 3+ Saves Factor"])
          * model.baseline("saves3", gk["p"], gk["po"], pd.Series("GK", index=gk.index), ghome))
    p6 = (gk["Player"].map(fb["F1 6+ Saves Factor"]).fillna(med["F1 6+ Saves Factor"])
          * model.baseline("saves6", gk["p"], gk["po"], pd.Series("GK", index=gk.index), ghome))
    pd.DataFrame({"Match": gk["Team"] + " v " + gk["opp"], "Date": EPOCH, "Team": gk["Team"],
                  "Goalkeeper": gk["Player"], "3+ Saves": _to_odds(p3), "6+ Saves": _to_odds(p6)}).to_csv(
        os.path.join(sb, "sportsbet_goalkeeper_saves_odds.csv"), index=False)

    # --- team markets: F1 only. The _f2 files are CLEARED (header-only) so F2 clean-sheet/team-goals
    # are model-derived off F1 rather than synthetic; real Betway F2 odds repopulate them when priced.
    cs1, tg1 = _team_block(gw1, EPOCH)
    cs1.to_csv(os.path.join(sb, "sportsbet_clean_sheet_odds.csv"), index=False)
    tg1.to_csv(os.path.join(sb, "sportsbet_team_goals_odds.csv"), index=False)
    cs1.iloc[0:0].to_csv(os.path.join(sb, "sportsbet_clean_sheet_odds_f2.csv"), index=False)
    tg1.iloc[0:0].to_csv(os.path.join(sb, "sportsbet_team_goals_odds_f2.csv"), index=False)

    # Provenance is the single source of truth for what's synthetic vs real (self-updating, never
    # goes stale) - no companion note file. betway.py / ladbrokes_cards.py flip each market to real
    # as it lands; `tools/odds_status.py` shows current state.
    provenance.reset_synthetic(a.gw, [
        "sportsbet_win_draw_win_odds.csv", "sportsbet_goalscorer_odds.csv", "sportsbet_two_goals_odds.csv",
        "sportsbet_assist_odds.csv", "sportsbet_booking_odds.csv", "sportsbet_goalkeeper_saves_odds.csv",
        "sportsbet_clean_sheet_odds.csv", "sportsbet_team_goals_odds.csv"])

    print(f"\nSynthetic GW{a.gw} F1 markets written to sportsbet/*.csv (F1 only - F2 derived off F1); "
          f"_f2 files cleared; provenance reset (all synthetic - see tools/odds_status.py). Run "
          f"tools/betway.py + tools/ladbrokes_cards.py to override with real odds.")


if __name__ == "__main__":
    main()
