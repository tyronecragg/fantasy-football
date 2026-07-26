import numpy as np
import pandas as pd

from fpl_pipeline import markets


def test_implied_basic_and_edges():
    odds = pd.Series([2.0, 0.0, np.nan, -1.0])
    out = markets.implied(odds, 1.05)
    assert np.isclose(out[0], 1 / 2.0 / 1.05)
    assert out[1:3].isna().all()          # zero and NaN odds -> NaN
    assert np.isclose(out[3], 1 / -1.0 / 1.05)  # negative passes through (garbage in)


def test_two_sided_normalises_margin():
    over, under = pd.Series([1.5]), pd.Series([2.5])
    out = markets.two_sided(over, under)
    assert np.isclose(out[0], (1 / 1.5) / (1 / 1.5 + 1 / 2.5))
    assert markets.two_sided(pd.Series([np.nan]), under).isna().all()


def test_f2_duplicate_guard():
    f1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    dup = f1.copy()
    distinct = f1.copy()
    distinct.loc[0, "b"] = 99
    assert len(markets._without_f1_duplicate(f1, dup, "x")) == 0
    assert len(markets._without_f1_duplicate(f1, distinct, "x")) == 2


def test_gk_saves_defaults():
    df = pd.DataFrame([["m", 1, "TeamA", "GK A", 1.62, np.nan],
                       ["m", 1, "TeamB", "GK B", np.nan, np.nan]])
    out = markets.gk_saves_market(df)
    assert np.isclose(out["prob3"][0], 1 / 1.62 / 1.05)
    assert out["prob3"][1] == 0.6   # IFERROR default
    assert (out["prob6"] == 0.0).all()
