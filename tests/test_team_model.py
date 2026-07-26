import numpy as np
import pandas as pd

from fpl_pipeline import team_model
from fpl_pipeline.names import apply_team_names


def _block(draw=None):
    data = {0: ["Alpha"], 1: ["Beta"], 2: [2.0], 3: [3.8]}
    df = pd.DataFrame(data)
    df.columns = ["home_team", "away_team", "home_win_odds", "away_win_odds"]
    if draw is not None:
        df["draw_odds"] = draw
    return df


def test_win_prob_legacy():
    teams = pd.Series(["Alpha", "Beta", "Gamma"])
    out = team_model._win_prob_lookup(teams, _block(), draw_aware=False)
    assert np.isclose(out[0], 1 / 2.0 / 1.03)
    assert np.isclose(out[1], 1 / 3.8 / 1.03)
    assert np.isnan(out[2])  # no match in block


def test_win_prob_draw_aware():
    teams = pd.Series(["Alpha", "Beta"])
    out = team_model._win_prob_lookup(teams, _block(draw=3.4), draw_aware=True)
    total = 1 / 2.0 + 1 / 3.4 + 1 / 3.8
    assert np.isclose(out[0], (1 / 2.0) / total)
    assert np.isclose(out[1], (1 / 3.8) / total)
    # probabilities now genuinely sum below 1 for the two sides
    assert out[0] + out[1] < 1


def test_win_prob_draw_aware_falls_back_without_draw_odds():
    teams = pd.Series(["Alpha"])
    missing = team_model._win_prob_lookup(teams, _block(draw=np.nan), draw_aware=True)
    assert np.isclose(missing[0], 1 / 2.0 / 1.03)  # per-row legacy fallback


def test_season_probs_shape_and_cleanup(inputs):
    season = team_model.season_probs(inputs)
    assert len(season) == 20
    assert set(apply_team_names(season["team_raw"])) == set(season["team"])
    valid = season["title"].notna()
    assert np.allclose(season.loc[valid, "title"], 1 / season.loc[valid, "title_odds"] / 1.08)
