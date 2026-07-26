"""Refit round-trip: fitting on data generated from the current coefficients must
reproduce the current model's *predictions*. (Raw coefficients are not identifiable —
the workbook's feature sets are collinear, e.g. diff = win - opp — so predictions are
the invariant, not coefficient values.)"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

from fpl_pipeline import model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("refit", os.path.join(ROOT, "tools", "refit_coefficients.py"))
refit = importlib.util.module_from_spec(spec)
sys.modules["refit"] = refit
spec.loader.exec_module(refit)


def _synthetic_history(n=600, seed=7):
    rng = np.random.default_rng(seed)
    win = pd.Series(rng.uniform(0.05, 0.85, n))
    opp = pd.Series(rng.uniform(0.05, 0.85, n))
    pos = pd.Series(rng.choice(["GK", "DEF", "MID", "FWD"], n))
    venue = pd.Series(rng.choice(["H", "A"], n))

    df = pd.DataFrame({"Player Name": [f"P{i}" for i in range(n)], "Position": pos,
                       "Team": "T", "F1 Opponent": "O", "F1 Venue": venue,
                       "F1 Win": win, "F1 Opponent Win": opp})
    for stat, target in refit.BASELINE_TARGETS.items():
        df[target] = model.baseline(stat, win, opp, pos, venue == "H")
    return df


def test_baseline_refit_reproduces_predictions(tmp_path, monkeypatch):
    df = _synthetic_history()
    path = str(tmp_path / "hist.csv")
    df.to_csv(path, index=False)
    monkeypatch.setattr(refit, "PLAYER_HISTORY", path)

    report = []
    fitted = refit.fit_baselines(report)

    feats = model._features(df["F1 Win"], df["F1 Opponent Win"],
                            df["Position"], df["F1 Venue"] == "H")
    for stat, target in refit.BASELINE_TARGETS.items():
        pred = sum(c * feats[k] for k, c in fitted[stat].items())
        assert np.allclose(pred, df[target], atol=1e-4), f"{stat} predictions diverged"
    assert all(status == "refit" for _, status, *_ in report)


def test_refit_dry_run_never_writes(tmp_path, monkeypatch, capsys):
    df = _synthetic_history(n=200)
    hist_path = str(tmp_path / "hist.csv")
    df.to_csv(hist_path, index=False)
    monkeypatch.setattr(refit, "PLAYER_HISTORY", hist_path)

    fx = pd.DataFrame({
        "home_team": ["A"] * 60, "away_team": ["B"] * 60,
        "home_win_odds": np.linspace(1.5, 4, 60), "away_win_odds": np.linspace(4, 1.5, 60),
        "home_title_odds": 10, "away_title_odds": 20, "home_relegation_odds": 100,
        "away_relegation_odds": 50, "home_top_6_odds": 3, "away_top_6_odds": 6,
    })
    fx_path = str(tmp_path / "fx.csv")
    fx.to_csv(fx_path, index=False)
    monkeypatch.setattr(refit, "FIXTURE_HISTORY", fx_path)

    from fpl_pipeline import config
    before = os.path.getmtime(config.COEFFICIENTS_JSON)
    refit.main(write=False)
    assert os.path.getmtime(config.COEFFICIENTS_JSON) == before
    assert not os.path.exists(model.WORKBOOK_COEFFICIENTS_JSON)
    assert "Dry run" in capsys.readouterr().out
