import numpy as np
import pandas as pd

from fpl_pipeline import model


def test_poisson_curves_properties():
    p1 = pd.Series(np.linspace(0.0, 0.95, 40))
    p2, p3 = model.poisson_score2(p1), model.poisson_score3(p1)
    assert p2[0] == 0.0 and p3[0] == 0.0
    assert (p2.diff().dropna() >= 0).all() and (p3.diff().dropna() >= 0).all()  # monotone
    assert (p2[1:] < p1[1:]).all() and (p3[1:] < p2[1:]).all()  # tail ordering
    assert model.poisson_score2(pd.Series([np.nan])).isna().all()


def test_ladders_match_workbook_steps():
    p = pd.Series([0.29, 0.3, 0.49, 0.56, 0.61, np.nan])
    assert model.ladder_score2(p).tolist()[:5] == [0.01, 0.03, 0.05, 0.1, 0.28]
    assert model.ladder_score3(p).tolist()[:5] == [0.0, 0.01, 0.01, 0.02, 0.11]
    assert model.ladder_score2(p).isna()[5] and model.ladder_score3(p).isna()[5]


def test_reconcile_win_draw():
    from fpl_pipeline import config
    # [0] over-certain pair (sum 1.3): pulled to the draw floor, ratio preserved (not zero-draw)
    # [1] valid, in-band pair (draw 0.3 <= even ceiling): left UNCHANGED
    # [2] ballooned draw (0.14/0.49 -> resid 0.37): clamped down into the band
    win = pd.Series([0.8, 0.3, 0.14])
    opp = pd.Series([0.5, 0.4, 0.49])
    w, o = model.reconcile_win_draw(win, opp)
    draw = 1.0 - w - o
    # [0] over-certain -> draw == floor, win/opp ratio preserved
    assert np.isclose(draw[0], config.DRAW_FLOOR)
    assert np.isclose(w[0] / o[0], 0.8 / 0.5)
    # [1] in-band -> untouched
    assert np.isclose(w[1], 0.3) and np.isclose(o[1], 0.4)
    # [2] draw clamped below its 0.37 residual, and to the decisiveness-aware ceiling
    r = 0.14 / (0.14 + 0.49)
    ceil = config.DRAW_CEIL_EVEN - config.DRAW_CEIL_SLOPE * abs(2 * r - 1)
    assert draw[2] < 0.37 and np.isclose(draw[2], ceil)
    assert np.isclose(w[2] / o[2], 0.14 / 0.49)   # relative strength preserved


def test_dc_probability_midpoint():
    out = model.dc_probability(pd.Series([10.0, np.nan]), sd=4.0, threshold=10)
    assert np.isclose(out[0], 0.5)
    assert np.isnan(out[1])


def test_baseline_features_cover_all_coefficients():
    feats = model._features(pd.Series([0.5]), pd.Series([0.3]),
                            pd.Series(["MID"]), pd.Series([True]))
    for stat, coefs in model.BASELINES.items():
        missing = set(coefs) - set(feats)
        assert not missing, f"{stat}: unknown features {missing}"


def test_opp_win_pred_clamped_at_zero():
    zeros = pd.Series([0.0])
    big = pd.Series([5.0])
    out = model.opp_win_pred(big, zeros, zeros, big, pd.Series([True]))
    assert (out >= 0).all()


def test_load_coefficients_reload():
    before = model.COEFS["_source"]
    model.load_coefficients()
    assert model.COEFS["_source"] == before


def test_xp_pre_concede_penalty_follows_fpl_rule_in_improved_mode():
    from fpl_pipeline import model
    pos = pd.Series(["DEF", "MID"])
    start = pd.Series([1.0, 1.0])
    stats = {k: pd.Series([0.0, 0.0]) for k in ("score1", "score2", "score3", "assist", "yellow",
                                                "clean_sheet", "saves3", "saves6", "dc_def", "dc_mid")}
    stats["concede2"] = pd.Series([0.5, 0.5])
    stats["concede4"] = pd.Series([0.1, 0.1])
    workbook = model.xp_pre(pos, start, stats)                                  # 2 - 2*(0.5+0.1)
    fpl = model.xp_pre(pos, start, stats, concede_points=model.CONCEDE_POINTS_FPL)  # 2 - 1*(0.5+0.1)
    assert abs(workbook[0] - 0.8) < 1e-9 and abs(fpl[0] - 1.4) < 1e-9
    assert workbook[1] == fpl[1] == 2.0                                          # MID: no concede penalty


def test_poisson_tail_shift_round_trips_and_extends():
    from fpl_pipeline import model
    p4 = pd.Series([0.35, 0.05, float("nan")])
    same = model.poisson_tail_shift(p4, 4, 4)
    assert abs(same[0] - 0.35) < 1e-6 and abs(same[1] - 0.05) < 1e-6 and pd.isna(same[2])
    p6 = model.poisson_tail_shift(p4, 4, 6)
    assert 0.05 < p6[0] < 0.15 and 0.001 < p6[1] < 0.01 and pd.isna(p6[2])


def test_xp_pre_forward_dc_and_concede6_are_improved_only():
    from fpl_pipeline import model
    pos = pd.Series(["FWD", "DEF"])
    start = pd.Series([1.0, 1.0])
    stats = {k: pd.Series([0.0, 0.0]) for k in ("score1", "score2", "score3", "assist", "yellow",
                                                "clean_sheet", "concede2", "concede4", "saves3",
                                                "saves6", "dc_def", "dc_mid")}
    base = model.xp_pre(pos, start, stats, concede_points=1.0)
    assert base.tolist() == [2.0, 2.0]
    extra = dict(stats, dc_fwd=pd.Series([0.05, 0.9]), concede6=pd.Series([0.9, 0.02]))
    out = model.xp_pre(pos, start, extra, concede_points=1.0)
    assert abs(out[0] - 2.1) < 1e-9     # forward: +2 x 0.05 DC, no concede penalty
    assert abs(out[1] - 1.98) < 1e-9    # defender: -1 x 0.02 six-conceded, no forward DC


def test_score_tail_is_expected_goals_beyond_a_hat_trick():
    from fpl_pipeline import model
    p1 = pd.Series([1 - np.exp(-1.0), 0.3, 0.0, float("nan")])   # lam = 1, ~0.36, 0, missing
    tail = model.score_tail(p1)
    assert abs(tail[0] - 0.0234) < 5e-4          # lam=1: 1 - (0.6321 + 0.2642 + 0.0803) = P(4+)+P(5+)+...
    assert 0 < tail[1] < 0.002 and tail[2] == 0.0 and tail[3] == 0.0
    pos, start = pd.Series(["FWD", "FWD"]), pd.Series([1.0, 1.0])
    stats = {k: pd.Series([0.0, 0.0]) for k in ("score1", "score2", "score3", "assist", "yellow",
                                                "clean_sheet", "concede2", "concede4", "saves3",
                                                "saves6", "dc_def", "dc_mid")}
    base = model.xp_pre(pos, start, stats)
    with_tail = model.xp_pre(pos, start, dict(stats, score_tail=pd.Series([0.02, 0.0])))
    assert abs((with_tail[0] - base[0]) - 0.08) < 1e-9 and with_tail[1] == base[1]   # 4 pts x 0.02


def test_bonus_keeper_reserve_comes_off_the_pot_and_outfield_weights_reshape_the_share():
    from fpl_pipeline import config, model
    team = pd.Series(["A", "A", "A"])
    pwin, ploss = pd.Series([0.5] * 3), pd.Series([0.3] * 3)
    pos = pd.Series(["GK", "MID", "FWD"])
    weight = pd.Series([1.0, 1.0, 1.0]) * pos.map(config.BONUS_POSITION_WEIGHT).fillna(0.0)
    gk = pd.Series([config.BONUS_GK_PER_START, 0.0, 0.0])
    out = model.bonus_points_odds(weight, team, pwin, ploss, reserve=gk)
    pot = 0.5 * 5.62 + 0.2 * 3.24 + 0.3 * 0.78
    assert out[0] == 0.0                                            # keeper takes no share
    assert abs(out.sum() - (pot - config.BONUS_GK_PER_START)) < 1e-9   # outfielders share the remainder
    assert out[2] > out[1]                                          # forward weighted above midfielder
