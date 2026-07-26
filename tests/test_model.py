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


def test_scale_win_pair():
    win, opp = pd.Series([0.8, 0.3]), pd.Series([0.5, 0.4])
    w, o = model.scale_win_pair(win, opp)
    assert np.isclose(w[0] + o[0], 1.0)          # over-certain pair scaled to sum 1
    assert np.isclose(w[0] / o[0], 0.8 / 0.5)    # ratio preserved
    assert w[1] == 0.3 and o[1] == 0.4           # valid pair untouched


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
