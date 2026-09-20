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
6. Win/draw reconciliation: win_pred/opp_win_pred run per team-row, so a fixture has two rows
   with two estimates of each outcome. _unify_match averages them into ONE shared prediction
   (both rows then agree); model.reconcile_win_draw then clamps the residual draw into a
   decisiveness-aware band (~27% even, tapering for mismatches) and re-splits by ratio.
7. Smooth score curves: modelled P(score 2+/3+) come from a Poisson-consistent curve
   on P(score 1+) (model.poisson_score2/3) instead of the workbook's step ladders —
   which include an exact `p == 0.3` float-equality branch. F1 Score 2+ stays pure
   odds; F1 Score 3+ (no market exists) uses the same smooth curve.
8. F7/F8 fixture projections: same machinery as F3-F6, extending the optimiser's
   horizon to eight gameweeks (start probabilities come from curated F7/F8 columns
   in starting_lineups.csv, falling back to the F6 belief if absent).
9. F2 score uses the generic factor x baseline instead of the workbook's
   Coefficients-sheet score model. Backtested (tools/backtest_projections.py): the
   sheet model's MAE is 0.063 even given the fixture's actual odds, vs 0.019 for the
   generic machinery — the factor is calibrated against the full baseline, so
   multiplying it into a different model family breaks its scale.
10. Persistence blend: modelled F2-F6 score/assist/saves probabilities are blended with
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


def _dc_table(dc_stats, params_row, improved=True):
    """Defensive Contribution sheet equivalent: probability of hitting the DC threshold.

    Improved mode SHRINKS each player's own DC RATE (dc90) toward the reliable-population average
    dc90 in proportion to his evidence, THEN converts to a probability: weight = nineties /
    config.DC_SHRINK_NINETIES, capped at 1: at gate G, one full match = (1/G) own + rest average, and
    G-or-more full matches = his own rate untouched. Under
    config.DC_SHRINK_MIN_NINETIES (0.65 nineties ~= 59 min) the weight is zero - cameos are not
    evidence - so those players get the average, as do players with no minutes at all. The average is
    the mean dc90 of the reliable (>= gate) players, recomputed each run so it tracks the current data.
    Blending the RATE (not the probability) means "credit his real minutes at his rate, fill the rest
    with average-quality minutes" - the interpretable minutes-topped-up-to-the-gate model.

    Parity keeps the workbook's hard cliff — own PROBABILITY at >= 4 nineties, otherwise the frozen
    params_row["average_dc90"] — so its output stays byte-identical."""
    df = dc_stats.copy()
    n = df["nineties"].fillna(0).astype(float)
    df["prob"] = model.dc_probability(df["dc90"], params_row["sd"], params_row["threshold"])
    if improved:
        reliable = n >= config.DC_SHRINK_NINETIES
        avg_dc90 = float(df.loc[reliable, "dc90"].mean()) if reliable.any() else float(df["dc90"].mean())
        w = (n / config.DC_SHRINK_NINETIES).clip(0.0, 1.0)
        w = w.where(n >= config.DC_SHRINK_MIN_NINETIES, 0.0)   # under ~59 min played: no evidence
        df["dc_weight"] = w
        blended = w * df["dc90"].fillna(avg_dc90) + (1.0 - w) * avg_dc90   # blend the RATE
        if getattr(config, "DC_MINUTES_PER_START", False) and "mps" in df.columns:
            # per-90 rate -> expected count in the minutes he actually plays when he starts
            blended = blended * df["mps"].astype(float).fillna(90.0) / 90.0
        df["rate_eff"] = blended                     # per-start expected count, before any fixture effect
        df["prob_filled"] = model.dc_probability(blended, params_row["sd"], params_row["threshold"])
    else:
        df["prob_filled"] = df["prob"].where(n >= 4, params_row["average_dc90"])
    return df


def _unify_match(win, opp, team, opponent):
    """Collapse a fixture's two team-rows into ONE shared prediction (improved mode).

    win_pred/opp_win_pred run per team-row, so a match carries two estimates of each outcome:
    P(team wins) is this row's win_pred AND the opponent row's opp_win_pred. Averaging the two
    (a) uses both models' information — measured lower 1X2 error vs market than either row alone —
    and (b) makes the rows agree: P(team wins) = mean(team.win_pred, opp.opp_win_pred) is symmetric,
    so after reconcile_win_draw both rows carry an identical win/draw/loss. Win/opp are constant
    within a team, so a team->value map suffices; falls back to the row's own value if the opponent
    is absent from the data (e.g. a not-yet-registered side)."""
    tw = win.groupby(team).first()      # team -> its own win_pred
    to = opp.groupby(team).first()      # team -> its own opp_win_pred
    uni_win = ((win + opponent.map(to)) / 2.0).fillna(win)   # + opponent's view of THIS team winning
    uni_opp = ((opp + opponent.map(tw)) / 2.0).fillna(opp)   # + opponent's own win = this team's opp
    return uni_win, uni_opp


def _season_lookup(master, keys, season):
    return (vlookup(keys, season, "team", "title"),
            vlookup(keys, season, "team", "relegation"),
            vlookup(keys, season, "team", "top6"))


def _assist2_ratio(mkts, floor_p1=0.02, min_pairs=20):
    """Measured P(2+ assist) / P(1+ assist) from players Betway priced for BOTH.

    Assists cluster far more than a Poisson tail predicts (real 2+ runs ~0.24x the 1+ chance
    and stays roughly flat, vs model.poisson_score2's ~0.03-0.13 that rises with the 1+ chance
    and under-states 2+ assists 2-3x). So missing 2+ assists are filled with this ratio, not the
    curve. Falls back to config.ASSIST2_RATIO when too few real pairs exist to measure.
    """
    a1, a2 = mkts.get("assist"), mkts.get("assist2")
    if a1 is None or a2 is None or getattr(a2, "empty", True):
        return config.ASSIST2_RATIO
    both = a1.merge(a2, on="player", suffixes=("_1", "_2"))
    ok = both["prob_1"] > floor_p1
    if ok.sum() < min_pairs:
        return config.ASSIST2_RATIO
    return float((both.loc[ok, "prob_2"] / both.loc[ok, "prob_1"]).median())


def build(roster, season, teamview, mkts, lineups, fallback, dc_stats, dc_params,
          improved=True, factor_history=None, gameweek=None, assist_real=True, rating=None):
    """rating: a fpl_pipeline.rating.FixtureRating (improved mode) - the F2-fallback / F3-F8 win
    and opponent-win source; None = the season-outrights Lasso (model.win_pred)."""
    if improved:
        lineups = normalize_start_probs(lineups)
    m = pd.DataFrame({
        "Player Name": roster["name"],
        "Position": roster["position"].astype(str),
        "Team": roster["team"],
        "Cost": roster["cost"],
    })
    # FPL's per-season player id, for robust WITHIN-season joins/archiving (ids are reassigned each
    # season, so it is not a cross-season key). Guarded: parity's workbook roster has no id, so the
    # column is simply absent there and the workbook-exact output is unchanged.
    if "player_id" in roster.columns:
        m.insert(0, "player_id", roster["player_id"].values)
    pos = m["Position"]
    name = m["Player Name"]
    team = m["Team"]
    # 2+ assists don't follow the Poisson tail (they cluster) — fill from this measured ratio
    assist2_ratio = _assist2_ratio(mkts) if improved else 0.0

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

    # Start probabilities: Starting Lineups cols C:H (+ F7/F8 when curated), missing
    # player -> 0. A lineups file that only carries F1-F6 (parity snapshot, older
    # files) falls back to the F6 belief for F7/F8.
    lineup_cols = list(lineups.columns)
    for k in range(1, config.HORIZON + 1):
        col_idx = min(1 + k, len(lineup_cols) - 1)   # past the last curated column, HOLD it forward (F8)
        m[f"_start{k}"] = vlookup(name, lineups, lineup_cols[0],
                                  lineup_cols[col_idx]).fillna(0.0)

    # Defensive contribution probabilities (position-gated, 0 for other positions)
    prm = dc_params.set_index("position")
    dc_def = _dc_table(dc_stats["DEF"], prm.loc["DEF"], improved)
    dc_mid = _dc_table(dc_stats["MID"], prm.loc["MID"], improved)
    m["F1 Defensive Contribution - DEF"] = vlookup(name, dc_def, "name", "prob_filled").where(pos == "DEF", 0.0)
    m["F1 Defensive Contribution - MID"] = vlookup(name, dc_mid, "name", "prob_filled").where(pos == "MID", 0.0)
    # Forwards score the 12-threshold DC too (FPL: "midfielders & forwards"); the workbook never
    # modelled it, so improved mode only. ~3% of 90-minute forward appearances in 2025-26.
    dc_fwd_on = improved and "FWD" in dc_stats and "FWD" in prm.index
    if dc_fwd_on:
        dc_fwd = _dc_table(dc_stats["FWD"], prm.loc["FWD"], improved)
        m["F1 Defensive Contribution - FWD"] = vlookup(name, dc_fwd, "name", "prob_filled").where(pos == "FWD", 0.0)

    def dc_fwd_stat(prefix):
        return {"dc_fwd": m[f"{prefix} Defensive Contribution - FWD"]} if dc_fwd_on else {}

    # Improved mode: DefCon per FIXTURE. Each player's effective per-start rate (own rate blended,
    # x minutes per start) is re-priced for every fixture with the heavy-favourite multiplier from
    # THAT fixture's win probability (config.DC_FAVOURITE_CURVE), then converted to a probability.
    # Parity keeps the F1 table value copied across the horizon.
    dc_tables = {"DEF": dc_def, "MID": dc_mid}
    if dc_fwd_on:
        dc_tables["FWD"] = dc_fwd
    if improved:
        m["_dc_rate"] = 0.0
        for _pos, _tab in dc_tables.items():
            m["_dc_rate"] = m["_dc_rate"].where(pos != _pos, vlookup(name, _tab, "name", "rate_eff"))

    def dc_fixture(prefix, win):
        """Write {prefix} Defensive Contribution - DEF/MID(/FWD) from _dc_rate x favourite multiplier."""
        if not improved:
            for _pos in dc_tables:
                m[f"{prefix} Defensive Contribution - {_pos}"] = m[f"F1 Defensive Contribution - {_pos}"]
            return
        mult = model.dc_favourite_multiplier(win, pos)
        for _pos in dc_tables:
            row = prm.loc[_pos]
            p = model.dc_probability(m["_dc_rate"] * mult, row["sd"], row["threshold"])
            m[f"{prefix} Defensive Contribution - {_pos}"] = p.where(pos == _pos, 0.0).fillna(0.0)

    def concede6_stat(prefix):
        return {"concede6": m[f"{prefix} Concede 6+ Goals"]} if improved else {}

    # ---- F1: real odds ----
    m["F1 Start"] = m["_start1"]
    m["F1 Win"] = vlookup(team, teamview, "team", "f1_win")
    m["F1 Opponent Win"] = vlookup(team, teamview, "team", "f1_opponent_win")
    if improved:
        dc_fixture("F1", m["F1 Win"])                 # heavy-favourite discount on this week's odds
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
    if improved:
        # 2+ assists: real Betway price where it exists, else the same Poisson tail that
        # fills 2+ goals. Lets expected assists count a player's occasional two-assist game.
        # Improved-mode only (like F7/F8) — parity reproduces the workbook's columns exactly.
        real_a2 = vlookup(name, mkts["assist2"], "player", "prob")
        m["F1 Assist 2+"] = real_a2.where(real_a2.notna(), assist2_ratio * m["F1 Assist"])
    m["F1 Yellow Card"] = vlookup(name, mkts["yellow"], "player", "prob")
    m["F1 Clean Sheet"] = vlookup(team, mkts["clean_sheet"], "team", "prob")
    m["F1 Concede 2+ Goals"] = vlookup(team, mkts["concede"], "team", "prob2")
    m["F1 Concede 4+ Goals"] = vlookup(team, mkts["concede"], "team", "prob4")
    if improved:
        # Third -1 at six conceded. No bookmaker ladder reaches 5.5 team goals, so extend the 4+
        # tail (real price where Betway lists Over 3.5, Poisson off over-1.5 otherwise) by Poisson.
        m["F1 Concede 6+ Goals"] = model.poisson_tail_shift(m["F1 Concede 4+ Goals"], 4, 6)
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
    def _median_with_history(stat, computed):
        """Trailing-median factor: this season's archived weekly factors + the current week
        (backtested better than single-week; also robust to a one-off outlier week, e.g. a
        premium creator's assist priced low for a single tough away fixture)."""
        hist = (factor_history or {}).get(stat)
        if not (improved and hist):
            return computed
        # assist joins the median-always stats (2026-09-01): single-week backtested marginally
        # sharper on real-odds weeks, but pins the whole F2-F8 assist line to ONE week's real
        # print - so a tough-fixture low (Palmer 6.7% away at Arsenal) halved his F2-F8 vs the
        # GW1/GW2 snapshots. The median over the real-odds archive weeks rejects that outlier.
        # F1 itself stays the real single-week odds (the vlookup below is untouched when real);
        # only the FORWARD factor is smoothed, exactly like score1/yellow.
        return pd.Series(
            [np.median(np.append(hist[p], c)) if (p in hist and np.isfinite(c)) else c
             for p, c in zip(name, computed)], index=computed.index)

    for col, prob_col, stat, fb_idx in factor_specs:
        computed = m[prob_col] / model.baseline(stat, m["F1 Win"], m["F1 Opponent Win"], pos, home1)
        computed = _median_with_history(stat, computed)
        fb = vlookup(name, fallback, fb_cols[0], fb_cols[fb_idx])
        m[col] = computed.where(have_odds, fb)

    m["Clean Sheet Factor"] = _median_with_history(
        "clean_sheet", m["F1 Clean Sheet"] / model.baseline(
            "clean_sheet", m["F1 Win"], m["F1 Opponent Win"], pos, home1))

    # When there is no real assist market, this week's F1 Assist came from a synthetic carry that
    # the longshot calibration shrinks (Haaland 0.135 -> 0.083). The Assist Factor above has now
    # been medianed against the real-odds archive weeks, so rebuild F1 Assist from that robust
    # factor x baseline - exactly as F2-F8 do - instead of the calibrated carry. Real assist odds
    # keep their single-week value untouched (byte-identical to before).
    if improved and not assist_real:
        m["F1 Assist"] = m["Assist Factor"] * model.baseline(
            "assist", m["F1 Win"], m["F1 Opponent Win"], pos, home1)
        clip01(["F1 Assist"])
        real_a2 = vlookup(name, mkts["assist2"], "player", "prob")
        m["F1 Assist 2+"] = real_a2.where(real_a2.notna(), assist2_ratio * m["F1 Assist"])

    def xp_block(prefix, start, stats):
        pre = model.xp_pre(pos, start, stats,
                           concede_points=model.CONCEDE_POINTS_FPL if improved
                           else model.CONCEDE_POINTS_WORKBOOK)
        m[f"{prefix} XP Pre"] = pre
        if improved and config.BONUS_ODDS_MODEL:
            # odds-anchored bonus: team pot from win/draw/loss, split by performance above the
            # 2-pt appearance floor. Bonus-prob column holds expected-bonus/2 so XP = pre + 2*col.
            # keepers: flat expected bonus per start (nothing priced predicts it - see config), taken
            # off the team pot; outfielders share the remainder by XP above the floor x position weight
            gk_bonus = (config.BONUS_GK_PER_START * start.fillna(0.0)).where(pos == "GK", 0.0)
            weight = (pre - 2.0 * start.fillna(0.0)).clip(lower=0.0)
            weight = weight * pos.map(config.BONUS_POSITION_WEIGHT).fillna(0.0)   # GK -> 0 (not in the share)
            wc = f"{prefix} Win" if f"{prefix} Win" in m else f"{prefix} Win Pred"          # F1/F2 odds, F3+ predicted
            oc = f"{prefix} Opponent Win" if f"{prefix} Opponent Win" in m else f"{prefix} Opponent Win Pred"
            bpts = model.bonus_points_odds(weight, team, m[wc], m[oc], reserve=gk_bonus) + gk_bonus
            m[f"{prefix} Expected Bonus"] = bpts                    # expected bonus POINTS (what it says)
            m[f"{prefix} Bonus Probability"] = bpts / 2.0           # legacy column: XP = pre + 2 x this
            m[f"{prefix} XP"] = pre + bpts
        else:
            bonus = model.bonus_probability(pre)
            m[f"{prefix} Bonus Probability"] = bonus
            m[f"{prefix} XP"] = model.xp_with_bonus(pre, bonus)

    xp_block("F1", m["F1 Start"], {
        "score1": m["F1 Score 1+"], "score2": m["F1 Score 2+"], "score3": m["F1 Score 3+"],
        **({"score_tail": model.score_tail(m["F1 Score 1+"])} if improved else {}),
        "assist": m["F1 Assist"], **({"assist2": m["F1 Assist 2+"]} if improved else {}), "yellow": m["F1 Yellow Card"], "clean_sheet": m["F1 Clean Sheet"],
        "concede2": m["F1 Concede 2+ Goals"], "concede4": m["F1 Concede 4+ Goals"],
        "saves3": m["F1 3+ Saves"], "saves6": m["F1 6+ Saves"],
        **({"saves_tail": model.saves_tail(m["F1 3+ Saves"], m["F1 6+ Saves"])} if improved else {}),
        "dc_def": m["F1 Defensive Contribution - DEF"], "dc_mid": m["F1 Defensive Contribution - MID"],
        **dc_fwd_stat("F1"), **concede6_stat("F1"),
    })
    m["F1 Pred XP"] = model.baseline("pred_xp", m["F1 Win"], m["F1 Opponent Win"], pos, home1)

    # Trained forward-projection models override factor x baseline for the defensive markets in
    # F3-F8 (the only components that beat the real pipeline value on walk-forward). Needs the
    # current gameweek + the on-disk archive (trailing form). Off (factor x baseline) for bare or
    # parity runs; but a --gw run that is meant to use them and CAN'T load them now FAILS loudly
    # rather than silently degrading (set config.USE_PROJECTION_MODEL=False to opt out on purpose).
    server = None
    if improved and config.USE_PROJECTION_MODEL and gameweek:
        try:
            from . import history, projection_serving
            _archive = pd.read_csv(history.PLAYER_HISTORY_CSV, low_memory=False)
            server = projection_serving.make_server(m, _archive, config.SEASON, gameweek, rating=rating)
        except Exception as exc:  # projection models are REQUIRED here - fail, don't silently degrade
            raise RuntimeError(
                f"projection-model serving is required for this --gw run but failed: {exc}. "
                f"Run with the venv (env/Scripts/python) so lightgbm etc. are available, or set "
                f"config.USE_PROJECTION_MODEL=False to deliberately accept factor x baseline."
            ) from exc

    # ---- F2: partial odds, otherwise factor x baseline ----
    m["F2 Start"] = m["_start2"]
    m["F2 Win"] = vlookup(team, teamview, "team", "f2_win")
    m["F2 Opponent Win"] = vlookup(team, teamview, "team", "f2_opponent_win")
    m["F2 Opponent"] = vlookup(team, teamview, "team", "f2_opponent")
    m["F2 Venue"] = vlookup(team, teamview, "team", "f2_venue")
    m["F2 Opponent Title"], m["F2 Opponent Relegation"], m["F2 Opponent Top 6"] = \
        _season_lookup(m, m["F2 Opponent"], season)
    home2 = m["F2 Venue"] == "H"

    def predicted_win(opp_col, home, weeks_ahead):
        """Win / opponent-win for a future fixture: the rolling rating when available (already
        symmetric across the two team-rows, so no unify/reconcile), else the outrights Lasso."""
        base_win = model.win_pred(
            m["Title"], m["Relegation"], m["Top 6"],
            m[f"{opp_col} Title"], m[f"{opp_col} Relegation"], m[f"{opp_col} Top 6"], home)
        base_opp = model.opp_win_pred(
            m["Title"], m["Relegation"], m[f"{opp_col} Title"], m[f"{opp_col} Relegation"], home)
        base_win, base_opp = base_win.clip(0.0, 1.0), base_opp.clip(0.0, 1.0)
        if improved:
            base_win, base_opp = _unify_match(base_win, base_opp, m["Team"], m[opp_col])
            base_win, base_opp = model.reconcile_win_draw(base_win, base_opp)
        if rating is None or not improved:
            return base_win, base_opp
        rw, ro = rating.win_probs(m["Team"], m[opp_col], home, weeks_ahead)
        return rw.where(rw.notna(), base_win), ro.where(ro.notna(), base_opp)

    if improved:
        no_odds = m["F2 Win"].isna()
        if no_odds.any():
            pred_win, pred_opp = predicted_win("F2 Opponent", home2, weeks_ahead=1)
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
    f2_yellow_raw = m["F1 Yellow Card Factor"] * model.baseline("yellow", w2, ow2, pos, home2)

    def apply_stacks(prefix, raw_cols, keep_mask, k):
        """Stacked stats: value = w*GBM + (1-w)*F1-blend of the raw factor x baseline. The GBM reads
        the raw baseline as its `predicted` feature, so the column must hold the RAW value when the
        server is called; rows in keep_mask (e.g. real F2 odds) keep what they have."""
        if server is None:
            return
        gated = {s for s, wk in config.PROJECTION_STACK_MIN_WEEKS.items() if server.weeks < wk}
        stacks = {s: v for s, v in config.PROJECTION_STACK.items() if s in server.models and s not in gated}
        if not stacks:
            return
        saved = {s: m[f"{prefix} {lab}"].copy() for s, (lab, _) in stacks.items()}
        for s, (lab, _) in stacks.items():
            m[f"{prefix} {lab}"] = raw_cols[s]                # raw baseline -> the model's `predicted`
        preds = server.predict_horizon(m, k)
        for s, (lab, wgt) in stacks.items():
            col = f"{prefix} {lab}"
            blend = raw_cols[s].clip(0.0, 1.0)
            if s in config.PROJECTION_BLEND:               # the deployed F1-blend of the baseline
                wb = config.PROJECTION_BLEND[s]
                blend = wb * blend + (1 - wb) * m[f"F1 {lab}"].fillna(blend)
            stacked = (wgt * preds[s] + (1 - wgt) * blend).clip(0.0, 1.0)
            m[col] = saved[s].where(keep_mask.get(s, pd.Series(False, index=m.index)), stacked)

    blend_with_f1("F2 Score 1+", "F1 Score 1+", "score1")
    m["F2 Assist"] = m["Assist Factor"] * model.baseline("assist", w2, ow2, pos, home2)
    f2_yellow_odds = vlookup(name, mkts["f2_yellow"], "player", "prob") if len(mkts["f2_yellow"]) else pd.Series(np.nan, index=m.index)
    m["F2 Yellow Card"] = f2_yellow_odds.where(f2_yellow_odds.notna(), f2_yellow_raw)
    # F2 projection model (horizon 1): fills the factor x baseline FALLBACK for the deployed
    # defensive markets where F2 has no real odds (real Betway odds are always kept). saves3 has no
    # F2 market at all, so it is model-driven whenever the server is active.
    preds2 = server.predict_horizon(m, 2) if server is not None else None
    m["F2 Clean Sheet"] = vlookup(team, mkts["f2_clean_sheet"], "team", "prob")
    m["F2 Concede 2+ Goals"] = vlookup(team, mkts["f2_concede"], "team", "prob2")
    m["F2 Concede 4+ Goals"] = vlookup(team, mkts["f2_concede"], "team", "prob4")
    if improved:
        for col, factor_col, stat in [
            ("F2 Clean Sheet", "Clean Sheet Factor", "clean_sheet"),
            ("F2 Concede 2+ Goals", "F1 Concede 2+ Goals Factor", "concede2"),
            ("F2 Concede 4+ Goals", "F1 Concede 4+ Goals Factor", "concede4"),
        ]:
            fb = (preds2[stat] if (preds2 is not None and stat in preds2)
                  else m[factor_col] * model.baseline(stat, w2, ow2, pos, home2))
            m[col] = m[col].where(m[col].notna(), fb)
    if preds2 is not None:
        m["F2 3+ Saves"] = preds2["saves3"]                     # no real F2 saves market
    else:
        m["F2 3+ Saves"] = m["F1 3+ Saves Factor"] * model.baseline("saves3", w2, ow2, pos, home2)
    m["F2 6+ Saves"] = m["F1 6+ Saves Factor"] * model.baseline("saves6", w2, ow2, pos, home2)
    dc_fixture("F2", w2)                              # F2 odds, or the predicted win prob when unpriced
    clip01(["F2 Assist", "F2 Yellow Card", "F2 Clean Sheet", "F2 Concede 2+ Goals",
            "F2 Concede 4+ Goals", "F2 3+ Saves", "F2 6+ Saves"])
    if improved:
        m["F2 Concede 6+ Goals"] = model.poisson_tail_shift(m["F2 Concede 4+ Goals"], 4, 6)
    blend_with_f1("F2 Assist", "F1 Assist", "assist")
    if improved:
        m["F2 Assist 2+"] = assist2_ratio * m["F2 Assist"]
    if server is None:                                 # model-driven F2 saves3 is used PURE (no blend)
        blend_with_f1("F2 3+ Saves", "F1 3+ Saves", "saves3")
    if improved:
        f2_score_raw = (m["Score 1+ Factor"] * model.baseline("score1", w2, ow2, pos, home2)).clip(0.0, 1.0)
        apply_stacks("F2", {"score1": f2_score_raw, "yellow": f2_yellow_raw.clip(0.0, 1.0)},
                     {"yellow": f2_yellow_odds.notna()}, 2)
    m["F2 Score 2+"], m["F2 Score 3+"] = score_curves(m["F2 Score 1+"])   # after any stack (both modes)

    xp_block("F2", m["F2 Start"], {
        "score1": m["F2 Score 1+"], "score2": m["F2 Score 2+"], "score3": m["F2 Score 3+"],
        **({"score_tail": model.score_tail(m["F2 Score 1+"])} if improved else {}),
        "assist": m["F2 Assist"], **({"assist2": m["F2 Assist 2+"]} if improved else {}), "yellow": m["F2 Yellow Card"], "clean_sheet": m["F2 Clean Sheet"],
        "concede2": m["F2 Concede 2+ Goals"], "concede4": m["F2 Concede 4+ Goals"],
        "saves3": m["F2 3+ Saves"], "saves6": m["F2 6+ Saves"],
        **({"saves_tail": model.saves_tail(m["F2 3+ Saves"], m["F2 6+ Saves"])} if improved else {}),
        "dc_def": m["F2 Defensive Contribution - DEF"], "dc_mid": m["F2 Defensive Contribution - MID"],
        **dc_fwd_stat("F2"), **concede6_stat("F2"),
    })

    # ---- F3..F6 (parity) / F3..F8 (improved): fully model-driven ----
    # F7/F8 exist only in improved mode: the workbook never computed them, and the
    # fixture data (opponents/venues through F8) has always been available. They give
    # the transfer optimiser a two-month horizon.
    for k in range(3, (config.HORIZON + 1) if improved else 7):
        p = f"F{k}"
        m[f"{p} Start"] = m[f"_start{k}"]
        m[f"{p} Opponent"] = vlookup(team, teamview, "team", f"f{k}_opponent")
        m[f"{p} Venue"] = vlookup(team, teamview, "team", f"f{k}_venue")
        m[f"{p} Opponent Title"], m[f"{p} Opponent Relegation"], m[f"{p} Opponent Top 6"] = \
            _season_lookup(m, m[f"{p} Opponent"], season)

        home = m[f"{p} Venue"] == "H"
        if improved:
            m[f"{p} Win Pred"], m[f"{p} Opponent Win Pred"] = predicted_win(f"{p} Opponent", home, weeks_ahead=k - 1)
        else:                                   # parity: the workbook's per-row Lasso, unclamped
            m[f"{p} Win Pred"] = model.win_pred(
                m["Title"], m["Relegation"], m["Top 6"],
                m[f"{p} Opponent Title"], m[f"{p} Opponent Relegation"], m[f"{p} Opponent Top 6"], home)
            m[f"{p} Opponent Win Pred"] = model.opp_win_pred(
                m["Title"], m["Relegation"],
                m[f"{p} Opponent Title"], m[f"{p} Opponent Relegation"], home)
            clip01([f"{p} Win Pred", f"{p} Opponent Win Pred"])
        m[f"{p} Diff"] = m[f"{p} Win Pred"] - m[f"{p} Opponent Win Pred"]

        w, ow = m[f"{p} Win Pred"], m[f"{p} Opponent Win Pred"]
        m[f"{p} Score 1+"] = m["Score 1+ Factor"] * model.baseline("score1", w, ow, pos, home)
        clip01([f"{p} Score 1+"])
        score_raw = m[f"{p} Score 1+"].copy()
        blend_with_f1(f"{p} Score 1+", "F1 Score 1+", "score1")
        m[f"{p} Assist"] = m["Assist Factor"] * model.baseline("assist", w, ow, pos, home)
        m[f"{p} Yellow Card"] = m["F1 Yellow Card Factor"] * model.baseline("yellow", w, ow, pos, home)
        yellow_raw = m[f"{p} Yellow Card"].clip(0.0, 1.0)
        m[f"{p} Clean Sheet"] = m["Clean Sheet Factor"] * model.baseline("clean_sheet", w, ow, pos, home)
        m[f"{p} Concede 2+ Goals"] = m["F1 Concede 2+ Goals Factor"] * model.baseline("concede2", w, ow, pos, home)
        m[f"{p} Concede 4+ Goals"] = m["F1 Concede 4+ Goals Factor"] * model.baseline("concede4", w, ow, pos, home)
        m[f"{p} 3+ Saves"] = m["F1 3+ Saves Factor"] * model.baseline("saves3", w, ow, pos, home)
        m[f"{p} 6+ Saves"] = m["F1 6+ Saves Factor"] * model.baseline("saves6", w, ow, pos, home)
        if server is not None:                         # model overrides factor x baseline (F3-F8)
            preds = server.predict_horizon(m, k)       # clean_sheet/concede2 pure; saves3 blended below
            for _stat, _label in config.PROJECTION_MODEL_STATS.items():
                m[f"{p} {_label}"] = preds[_stat].values
            apply_stacks(p, {"score1": score_raw, "yellow": yellow_raw}, {}, k)
        m[f"{p} Score 2+"], m[f"{p} Score 3+"] = score_curves(m[f"{p} Score 1+"])
        dc_fixture(p, w)                              # this fixture's predicted win prob
        clip01([f"{p} Assist", f"{p} Yellow Card", f"{p} Clean Sheet", f"{p} Concede 2+ Goals",
                f"{p} Concede 4+ Goals", f"{p} 3+ Saves", f"{p} 6+ Saves"])
        if improved:
            m[f"{p} Concede 6+ Goals"] = model.poisson_tail_shift(m[f"{p} Concede 4+ Goals"], 4, 6)
        blend_with_f1(f"{p} Assist", "F1 Assist", "assist")
        if improved:
            m[f"{p} Assist 2+"] = assist2_ratio * m[f"{p} Assist"]
        if server is None:                             # model-driven saves3 is used PURE (no blend)
            blend_with_f1(f"{p} 3+ Saves", "F1 3+ Saves", "saves3")

        xp_block(p, m[f"{p} Start"], {
            "score1": m[f"{p} Score 1+"], "score2": m[f"{p} Score 2+"], "score3": m[f"{p} Score 3+"],
            **({"score_tail": model.score_tail(m[f"{p} Score 1+"])} if improved else {}),
            "assist": m[f"{p} Assist"], **({"assist2": m[f"{p} Assist 2+"]} if improved else {}), "yellow": m[f"{p} Yellow Card"], "clean_sheet": m[f"{p} Clean Sheet"],
            "concede2": m[f"{p} Concede 2+ Goals"], "concede4": m[f"{p} Concede 4+ Goals"],
            "saves3": m[f"{p} 3+ Saves"], "saves6": m[f"{p} 6+ Saves"],
            **({"saves_tail": model.saves_tail(m[f"{p} 3+ Saves"], m[f"{p} 6+ Saves"])} if improved else {}),
            "dc_def": m[f"{p} Defensive Contribution - DEF"], "dc_mid": m[f"{p} Defensive Contribution - MID"],
            **dc_fwd_stat(p), **concede6_stat(p),
        })

    return m.drop(columns=[c for c in m.columns if c.startswith("_start") or c == "_dc_rate"])
