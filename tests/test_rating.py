import json

import numpy as np
import pandas as pd

from fpl_pipeline import config, rating


def _synthetic_matches(strength, hfa=0.3, seed=0):
    """Every pair home and away, prices generated from known strengths (no noise)."""
    rng = np.random.default_rng(seed)
    rows = []
    teams = list(strength)
    day = pd.Timestamp("2026-01-01")
    for i, h in enumerate(teams):
        for a in teams:
            if h == a:
                continue
            y = strength[h] - strength[a] + hfa
            ph = 1 / (1 + np.exp(-y)) * 0.75            # leave room for the draw
            pa = 0.75 - ph
            rows.append({"season": "2025-2026", "date": day + pd.Timedelta(days=int(rng.integers(0, 60))),
                         "HomeTeam": h, "AwayTeam": a, "p_home": ph, "p_draw": 0.25, "p_away": pa,
                         "y": np.log(ph / pa)})
    return pd.DataFrame(rows)


def test_solve_recovers_strengths_and_home_advantage():
    strength = {"A": 1.0, "B": 0.5, "C": 0.0, "D": -0.5, "E": -1.0}
    m = _synthetic_matches(strength)
    r, hfa = rating.solve(m, list(strength), np.ones(len(m)))
    assert abs(hfa - 0.3) < 1e-3
    for t, v in strength.items():
        assert abs(r[t] - v) < 1e-3


def test_promoted_side_takes_the_prior_level_and_anchor_is_current_league():
    strength = {"A": 1.0, "B": 0.5, "C": 0.0, "D": -0.5, "E": -1.0}
    m = _synthetic_matches(strength)
    pseudo = {"NEW": (-0.8, 2.0)}
    r, _ = rating.solve(m, list(strength) + ["NEW"], np.ones(len(m)), pseudo, anchor=["A", "B", "C", "D", "NEW"])
    assert abs(r["NEW"] + 0.8) < 0.05                 # pinned by its pseudo-observation
    assert abs(r[["A", "B", "C", "D", "NEW"]].sum()) < 1e-6   # anchored league sums to zero


def test_maps_predict_sum_to_one_and_are_venue_symmetric(tmp_path, monkeypatch):
    doc = {"horizons": {str(k): {"coef": [[1.5, 0.0, 0.0], [0.0, -0.5, 0.0], [-1.5, 0.0, 0.0]],
                                 "intercept": [0.2, 0.0, -0.2]} for k in range(1, 9)}}
    path = tmp_path / "maps.json"
    path.write_text(json.dumps(doc))
    monkeypatch.setattr(config, "RATING_MAPS_JSON", str(path))
    maps = rating.RatingMaps.load()
    p = maps.predict([0.5, -0.5, 0.0], 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert p[0, 0] > p[0, 2] and p[1, 2] > p[1, 0]     # stronger side favoured either way round
    fr = rating.FixtureRating(pd.Series({"X": 0.5, "Y": 0.0}), 0.2, 30.0, maps, pd.Timestamp("2026-09-01"), "2026-2027")
    w_home, o_home = fr.win_probs(pd.Series(["X"]), pd.Series(["Y"]), pd.Series([True]), 2)
    w_away, o_away = fr.win_probs(pd.Series(["Y"]), pd.Series(["X"]), pd.Series([False]), 2)
    assert abs(w_home[0] - o_away[0]) < 1e-12 and abs(o_home[0] - w_away[0]) < 1e-12   # same match, both rows
    w_unknown, _ = fr.win_probs(pd.Series(["X"]), pd.Series(["Nobody"]), pd.Series([True]), 2)
    assert np.isnan(w_unknown[0])                          # caller falls back to the incumbent


def test_live_maps_file_and_rating_load():
    """The shipped maps fit and the live rating loads from the cached prices (no network needed)."""
    maps = rating.RatingMaps.load()
    assert set(maps.h) == set(rating.HORIZONS)
    fr = rating.load(cutoff="2026-09-02")
    assert fr is not None and fr.weight > 10
    w, o = fr.win_probs(pd.Series(["Arsenal"]), pd.Series(["Hull City"]), pd.Series([True]), 1)
    assert w[0] > 0.6 and o[0] < 0.2
