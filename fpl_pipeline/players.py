"""Build the master players DataFrame — the Python replica of the Players sheet.

Column names deliberately match the workbook headers exactly so the parity harness can
compare by name. Known deliberate deviations from the workbook:

1. F2 Score 1+ row 2: the workbook's AW2 is a leftover F1 VLOOKUP; we apply the
   Coefficients model uniformly (workbook rows 3+ behaviour).
2. Fallback-factor lookups: the workbook uses approximate-match VLOOKUP on an unsorted
   sheet (undefined results); we use exact name match.
3. Opponent/venue lookups that hit a blank cell: Excel VLOOKUP renders blank as 0; the
   pipeline keeps NaN (validate.py treats that pattern as a known deviation).
Improved mode (build(..., improved=True); parity mode = improved=False is workbook-exact):

4. F2 model fallback: when F2 match odds are missing (e.g. gameweek 1, when bookmakers
   have only priced one gameweek), F2 win/opponent-win are predicted from season odds
   exactly like F3-F6, and clean-sheet/concede probabilities fall back to
   factor x baseline. Real F2 odds always take precedence when present.
5. Probability clamping: the workbook's regressions are unbounded linear models (the
   win prediction has no clamp, the opponent-win prediction is only clamped at 0, and
   factor x baseline stats can leave [0, 1]). Every modelled probability is clipped to
   [0, 1] before it feeds the score curves and XP scoring. Odds-derived probabilities
   are untouched.
6. Win-pair scaling: independently predicted win + opponent-win pairs that sum above 1
   are scaled down proportionally (model.scale_win_pair).
7. Smooth score curves: modelled P(score 2+/3+) come from a Poisson-consistent curve
   on P(score 1+) (model.poisson_score2/3) instead of the workbook's step ladders —
   which include an exact `p == 0.3` float-equality branch. F1 Score 2+ stays pure
   odds; F1 Score 3+ (no market exists) uses the same smooth curve.
8. F2 score uses the generic factor x baseline instead of the workbook's
   Coefficients-sheet score model. Backtested (tools/backtest_projections.py): the
   sheet model's MAE is 0.063 even given the fixture's actual odds, vs 0.019 for the
   generic machinery — the factor is calibrated against the full baseline, so
   multiplying it into a different model family breaks its scale.
9. Persistence blend: modelled F2-F6 score/assist/saves probabilities are blended with
   the player's current F1 odds-implied probability at backtested weights
   (config.PROJECTION_BLEND). Other stats backtested best as pure model.
"""
import warnings

import numpy as np
import pandas as pd

from . import config

# Columns are appended one at a time to mirror the sheet layout; the frame is small.
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from . import model
from .io_utils import vlookup


def normalize_start_probs(lineups, target=11.0):
    """Coherence invariant: each team's start probabilities sum to exactly 11 per
    fixture. Applied at CONSUMPTION (improved-mode builds and odds synthesis) — the
    stored starting_lineups.csv keeps raw beliefs so declarations like 'Kinský 1.0'
    are never degraded by repeated re-normalization.

    Over 11: the excess is squeezed out of uncertain players only (certainty = 1.0
    survives; if 11+ players are certain, everyone scales — a contradiction).
    Under 11: water-filled up with individual probabilities capped at 1.0."""
    lineups = lineups.copy()
    prob_cols = [c for c in lineups.columns if c.startswith("F") and c[1:].isdigit()]
    for team, idx in lineups.groupby("Team").groups.items():
        for col in prob_cols:
            p = lineups.loc[idx, col].astype(float).clip(0.0, 1.0)
            total = p.sum()
            if total <= 0:
                continue
            if total >= target:
                certain = p >= 0.999
                reducible = p[~certain].sum()
                need = target - p[certain].sum()
                if need >= 0 and reducible > 0:
                    p.loc[~certain] *= need / reducible
                else:
                    p = p * (target / total)
            else:
                capped = pd.Series(False, index=p.index)
                for _ in range(20):
                    free = ~capped
                    free_sum = p[free].sum()
                    if free_sum <= 0:
                        break
                    p.loc[free] *= (target - p[capped].sum()) / free_sum
                    newly = free & (p >= 1.0)
                    if not newly.any():
                        break
                    p.loc[newly] = 1.0
                    capped |= newly
            lineups.loc[idx, col] = p
    return lineups


def _dc_table(dc_stats, params_row):
    """Defensive Contribution sheet equivalent: probability of hitting the DC threshold,
    falling back to the position average for players with under 4 full matches."""
    df = dc_stats.copy()
    df["prob"] = model.dc_probability(df["dc90"], params_row["sd"], params_row["threshold"])
    df["prob_filled"] = df["prob"].where(df["nineties"].fillna(0) >= 4, params_row["average_dc90"])
    return df


def _season_lookup(master, keys, season):
    return (vlookup(keys, season, "team", "title"),
            vlookup(keys, season, "team", "relegation"),
            vlookup(keys, season, "team", "top6"))


def build(roster, season, teamview, mkts, lineups, fallback, dc_stats, dc_params,
          improved=True):
    if improved:
        lineups = normalize_start_probs(lineups)
    m = pd.DataFrame({
        "Player Name": roster["name"],
        "Position": roster["position"].astype(str),
        "Team": roster["team"],
        "Cost": roster["cost"],
    })
    pos = m["Position"]
    name = m["Player Name"]
    team = m["Team"]

    def clip01(cols):
        """Clamp modelled probability columns to [0, 1] (improved mode only)."""
        if improved:
            for c in cols:
                m[c] = m[c].clip(0.0, 1.0)

    def score_curves(p1):
        """P(score 2+), P(score 3+) from P(score 1+): Poisson curve in improved mode,
        the workbook's step ladders in parity mode."""
        if improved:
            return model.poisson_score2(p1), model.poisson_score3(p1)
        return model.ladder_score2(p1), model.ladder_score3(p1)

    def blend_with_f1(col, f1_col, stat):
        """Blend a modelled probability with the player's current F1 odds-implied one
        (improved mode; players without F1 odds keep the pure model value)."""
        w = config.PROJECTION_BLEND.get(stat)
        if improved and w is not None:
            f1 = m[f1_col]
            m[col] = (w * m[col] + (1 - w) * f1).where(f1.notna(), m[col])

    # Season probabilities (Overall Odds)
    m["Title"], m["Relegation"], m["Top 6"] = _season_lookup(m, team, season)

    # Start probabilities: Starting Lineups cols C:H, missing player -> 0
    lineup_cols = list(lineups.columns)
    for k in range(1, 7):
        m[f"_start{k}"] = vlookup(name, lineups, lineup_cols[0], lineup_cols[1 + k]).fillna(0.0)

    # Defensive contribution probabilities (position-gated, 0 for other positions)
    prm = dc_params.set_index("position")
    dc_def = _dc_table(dc_stats["DEF"], prm.loc["DEF"])
    dc_mid = _dc_table(dc_stats["MID"], prm.loc["MID"])
    m["F1 Defensive Contribution - DEF"] = vlookup(name, dc_def, "name", "prob_filled").where(pos == "DEF", 0.0)
    m["F1 Defensive Contribution - MID"] = vlookup(name, dc_mid, "name", "prob_filled").where(pos == "MID", 0.0)

    # ---- F1: real odds ----
    m["F1 Start"] = m["_start1"]
    m["F1 Win"] = vlookup(team, teamview, "team", "f1_win")
    m["F1 Opponent Win"] = vlookup(team, teamview, "team", "f1_opponent_win")
    m["F1 Diff"] = m["F1 Win"] - m["F1 Opponent Win"]
    m["F1 Opponent"] = vlookup(team, teamview, "team", "f1_opponent")
    m["F1 Venue"] = vlookup(team, teamview, "team", "f1_venue")
    m["F1 Opponent Title"], m["F1 Opponent Relegation"], m["F1 Opponent Top 6"] = \
        _season_lookup(m, m["F1 Opponent"], season)

    m["F1 Score 1+"] = vlookup(name, mkts["score1"], "player", "prob")
    m["F1 Score 2+"] = vlookup(name, mkts["score2"], "player", "prob")
    m["F1 Score 3+"] = (model.poisson_score3(m["F1 Score 1+"]) if improved
                        else model.ladder_score3(m["F1 Score 1+"]))
    m["F1 Assist"] = vlookup(name, mkts["assist"], "player", "prob")
    m["F1 Yellow Card"] = vlookup(name, mkts["yellow"], "player", "prob")
    m["F1 Clean Sheet"] = vlookup(team, mkts["clean_sheet"], "team", "prob")
    m["F1 Concede 2+ Goals"] = vlookup(team, mkts["concede"], "team", "prob2")
    m["F1 Concede 4+ Goals"] = vlookup(team, mkts["concede"], "team", "prob4")
    m["F1 3+ Saves"] = vlookup(team, mkts["gk_saves"], "team", "prob3").where(pos == "GK")
    m["F1 6+ Saves"] = vlookup(team, mkts["gk_saves"], "team", "prob6").where(pos == "GK")

    # ---- Factors: actual F1 probability / regression baseline, else fallback sheet ----
    home1 = m["F1 Venue"] == "H"
    have_odds = m["F1 Win"].notna()
    fb_cols = list(fallback.columns)

    factor_specs = [
        ("Score 1+ Factor", "F1 Score 1+", "score1", 1),
        ("Assist Factor", "F1 Assist", "assist", 2),
        ("F1 Yellow Card Factor", "F1 Yellow Card", "yellow", 3),
        ("F1 Concede 2+ Goals Factor", "F1 Concede 2+ Goals", "concede2", 4),
        ("F1 Concede 4+ Goals Factor", "F1 Concede 4+ Goals", "concede4", 5),
        ("F1 3+ Saves Factor", "F1 3+ Saves", "saves3", 6),
        ("F1 6+ Saves Factor", "F1 6+ Saves", "saves6", 7),
    ]
    for col, prob_col, stat, fb_idx in factor_specs:
        computed = m[prob_col] / model.baseline(stat, m["F1 Win"], m["F1 Opponent Win"], pos, home1)
        fb = vlookup(name, fallback, fb_cols[0], fb_cols[fb_idx])
        m[col] = computed.where(have_odds, fb)

    m["Clean Sheet Factor"] = m["F1 Clean Sheet"] / model.baseline(
        "clean_sheet", m["F1 Win"], m["F1 Opponent Win"], pos, home1)

    def xp_block(prefix, start, stats):
        pre = model.xp_pre(pos, start, stats)
        bonus = model.bonus_probability(pre)
        m[f"{prefix} XP Pre"] = pre
        m[f"{prefix} Bonus Probability"] = bonus
        m[f"{prefix} XP"] = model.xp_with_bonus(pre, bonus)

    xp_block("F1", m["F1 Start"], {
        "score1": m["F1 Score 1+"], "score2": m["F1 Score 2+"], "score3": m["F1 Score 3+"],
        "assist": m["F1 Assist"], "yellow": m["F1 Yellow Card"], "clean_sheet": m["F1 Clean Sheet"],
        "concede2": m["F1 Concede 2+ Goals"], "concede4": m["F1 Concede 4+ Goals"],
        "saves3": m["F1 3+ Saves"], "saves6": m["F1 6+ Saves"],
        "dc_def": m["F1 Defensive Contribution - DEF"], "dc_mid": m["F1 Defensive Contribution - MID"],
    })
    m["F1 Pred XP"] = model.baseline("pred_xp", m["F1 Win"], m["F1 Opponent Win"], pos, home1)

    # ---- F2: partial odds, otherwise factor x baseline ----
    m["F2 Start"] = m["_start2"]
    m["F2 Win"] = vlookup(team, teamview, "team", "f2_win")
    m["F2 Opponent Win"] = vlookup(team, teamview, "team", "f2_opponent_win")
    m["F2 Opponent"] = vlookup(team, teamview, "team", "f2_opponent")
    m["F2 Venue"] = vlookup(team, teamview, "team", "f2_venue")
    m["F2 Opponent Title"], m["F2 Opponent Relegation"], m["F2 Opponent Top 6"] = \
        _season_lookup(m, m["F2 Opponent"], season)
    home2 = m["F2 Venue"] == "H"

    if improved:
        no_odds = m["F2 Win"].isna()
        if no_odds.any():
            pred_win = model.win_pred(
                m["Title"], m["Relegation"], m["Top 6"],
                m["F2 Opponent Title"], m["F2 Opponent Relegation"], m["F2 Opponent Top 6"], home2)
            pred_opp = model.opp_win_pred(
                m["Title"], m["Relegation"],
                m["F2 Opponent Title"], m["F2 Opponent Relegation"], home2)
            pred_win, pred_opp = model.scale_win_pair(pred_win.clip(0.0, 1.0), pred_opp.clip(0.0, 1.0))
            m["F2 Win"] = m["F2 Win"].where(~no_odds, pred_win)
            m["F2 Opponent Win"] = m["F2 Opponent Win"].where(~no_odds, pred_opp)
            n_pred = int((no_odds & m["F2 Win"].notna()).sum())
            if n_pred:
                print(f"  F2 model fallback: no F2 match odds - predicted win probabilities for {n_pred} players")

    m["F2 Diff"] = m["F2 Win"] - m["F2 Opponent Win"]
    w2, ow2 = m["F2 Win"], m["F2 Opponent Win"]
    if improved:
        m["F2 Score 1+"] = m["Score 1+ Factor"] * model.baseline("score1", w2, ow2, pos, home2)
    else:
        m["F2 Score 1+"] = model.f2_score1(m["Score 1+ Factor"], w2, ow2, home2)
    clip01(["F2 Score 1+"])
    blend_with_f1("F2 Score 1+", "F1 Score 1+", "score1")
    m["F2 Score 2+"], m["F2 Score 3+"] = score_curves(m["F2 Score 1+"])
    m["F2 Assist"] = m["Assist Factor"] * model.baseline("assist", w2, ow2, pos, home2)
    f2_yellow_odds = vlookup(name, mkts["f2_yellow"], "player", "prob") if len(mkts["f2_yellow"]) else pd.Series(np.nan, index=m.index)
    m["F2 Yellow Card"] = f2_yellow_odds.where(
        f2_yellow_odds.notna(),
        m["F1 Yellow Card Factor"] * model.baseline("yellow", w2, ow2, pos, home2))
    m["F2 Clean Sheet"] = vlookup(team, mkts["f2_clean_sheet"], "team", "prob")
    m["F2 Concede 2+ Goals"] = vlookup(team, mkts["f2_concede"], "team", "prob2")
    m["F2 Concede 4+ Goals"] = vlookup(team, mkts["f2_concede"], "team", "prob4")
    if improved:
        for col, factor_col, stat in [
            ("F2 Clean Sheet", "Clean Sheet Factor", "clean_sheet"),
            ("F2 Concede 2+ Goals", "F1 Concede 2+ Goals Factor", "concede2"),
            ("F2 Concede 4+ Goals", "F1 Concede 4+ Goals Factor", "concede4"),
        ]:
            modelled = m[factor_col] * model.baseline(stat, w2, ow2, pos, home2)
            m[col] = m[col].where(m[col].notna(), modelled)
    m["F2 3+ Saves"] = m["F1 3+ Saves Factor"] * model.baseline("saves3", w2, ow2, pos, home2)
    m["F2 6+ Saves"] = m["F1 6+ Saves Factor"] * model.baseline("saves6", w2, ow2, pos, home2)
    m["F2 Defensive Contribution - DEF"] = m["F1 Defensive Contribution - DEF"]
    m["F2 Defensive Contribution - MID"] = m["F1 Defensive Contribution - MID"]
    clip01(["F2 Assist", "F2 Yellow Card", "F2 Clean Sheet", "F2 Concede 2+ Goals",
            "F2 Concede 4+ Goals", "F2 3+ Saves", "F2 6+ Saves"])
    blend_with_f1("F2 Assist", "F1 Assist", "assist")
    blend_with_f1("F2 3+ Saves", "F1 3+ Saves", "saves3")

    xp_block("F2", m["F2 Start"], {
        "score1": m["F2 Score 1+"], "score2": m["F2 Score 2+"], "score3": m["F2 Score 3+"],
        "assist": m["F2 Assist"], "yellow": m["F2 Yellow Card"], "clean_sheet": m["F2 Clean Sheet"],
        "concede2": m["F2 Concede 2+ Goals"], "concede4": m["F2 Concede 4+ Goals"],
        "saves3": m["F2 3+ Saves"], "saves6": m["F2 6+ Saves"],
        "dc_def": m["F2 Defensive Contribution - DEF"], "dc_mid": m["F2 Defensive Contribution - MID"],
    })

    # ---- F3..F6: fully model-driven ----
    for k in range(3, 7):
        p = f"F{k}"
        m[f"{p} Start"] = m[f"_start{k}"]
        m[f"{p} Opponent"] = vlookup(team, teamview, "team", f"f{k}_opponent")
        m[f"{p} Venue"] = vlookup(team, teamview, "team", f"f{k}_venue")
        m[f"{p} Opponent Title"], m[f"{p} Opponent Relegation"], m[f"{p} Opponent Top 6"] = \
            _season_lookup(m, m[f"{p} Opponent"], season)

        home = m[f"{p} Venue"] == "H"
        m[f"{p} Win Pred"] = model.win_pred(
            m["Title"], m["Relegation"], m["Top 6"],
            m[f"{p} Opponent Title"], m[f"{p} Opponent Relegation"], m[f"{p} Opponent Top 6"], home)
        m[f"{p} Opponent Win Pred"] = model.opp_win_pred(
            m["Title"], m["Relegation"],
            m[f"{p} Opponent Title"], m[f"{p} Opponent Relegation"], home)
        clip01([f"{p} Win Pred", f"{p} Opponent Win Pred"])
        if improved:
            m[f"{p} Win Pred"], m[f"{p} Opponent Win Pred"] = model.scale_win_pair(
                m[f"{p} Win Pred"], m[f"{p} Opponent Win Pred"])
        m[f"{p} Diff"] = m[f"{p} Win Pred"] - m[f"{p} Opponent Win Pred"]

        w, ow = m[f"{p} Win Pred"], m[f"{p} Opponent Win Pred"]
        m[f"{p} Score 1+"] = m["Score 1+ Factor"] * model.baseline("score1", w, ow, pos, home)
        clip01([f"{p} Score 1+"])
        blend_with_f1(f"{p} Score 1+", "F1 Score 1+", "score1")
        m[f"{p} Score 2+"], m[f"{p} Score 3+"] = score_curves(m[f"{p} Score 1+"])
        m[f"{p} Assist"] = m["Assist Factor"] * model.baseline("assist", w, ow, pos, home)
        m[f"{p} Yellow Card"] = m["F1 Yellow Card Factor"] * model.baseline("yellow", w, ow, pos, home)
        m[f"{p} Clean Sheet"] = m["Clean Sheet Factor"] * model.baseline("clean_sheet", w, ow, pos, home)
        m[f"{p} Concede 2+ Goals"] = m["F1 Concede 2+ Goals Factor"] * model.baseline("concede2", w, ow, pos, home)
        m[f"{p} Concede 4+ Goals"] = m["F1 Concede 4+ Goals Factor"] * model.baseline("concede4", w, ow, pos, home)
        m[f"{p} 3+ Saves"] = m["F1 3+ Saves Factor"] * model.baseline("saves3", w, ow, pos, home)
        m[f"{p} 6+ Saves"] = m["F1 6+ Saves Factor"] * model.baseline("saves6", w, ow, pos, home)
        m[f"{p} Defensive Contribution - DEF"] = m["F1 Defensive Contribution - DEF"]
        m[f"{p} Defensive Contribution - MID"] = m["F1 Defensive Contribution - MID"]
        clip01([f"{p} Assist", f"{p} Yellow Card", f"{p} Clean Sheet", f"{p} Concede 2+ Goals",
                f"{p} Concede 4+ Goals", f"{p} 3+ Saves", f"{p} 6+ Saves"])
        blend_with_f1(f"{p} Assist", "F1 Assist", "assist")
        blend_with_f1(f"{p} 3+ Saves", "F1 3+ Saves", "saves3")

        xp_block(p, m[f"{p} Start"], {
            "score1": m[f"{p} Score 1+"], "score2": m[f"{p} Score 2+"], "score3": m[f"{p} Score 3+"],
            "assist": m[f"{p} Assist"], "yellow": m[f"{p} Yellow Card"], "clean_sheet": m[f"{p} Clean Sheet"],
            "concede2": m[f"{p} Concede 2+ Goals"], "concede4": m[f"{p} Concede 4+ Goals"],
            "saves3": m[f"{p} 3+ Saves"], "saves6": m[f"{p} 6+ Saves"],
            "dc_def": m[f"{p} Defensive Contribution - DEF"], "dc_mid": m[f"{p} Defensive Contribution - MID"],
        })

    weights = model.COEFS["total_xp_weights"]
    m["Total XP"] = sum(wt * m[f"F{k} XP"] for k, wt in zip(range(1, 7), weights))

    return m.drop(columns=[c for c in m.columns if c.startswith("_start")])
