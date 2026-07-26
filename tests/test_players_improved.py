"""End-to-end improved-mode behaviour on a synthetic GW1 state: GW2 opponents known
(reverse fixtures) but no GW2 match odds — the exact single-gameweek scenario."""
import numpy as np
import pandas as pd
import pytest

from fpl_pipeline import markets, model, players, team_model

PROB_COLS = [f"{p} {s}" for p in ("F2", "F3", "F4", "F5", "F6")
             for s in ("Score 1+", "Score 2+", "Score 3+", "Assist", "Yellow Card",
                       "Clean Sheet", "Concede 2+ Goals", "Concede 4+ Goals",
                       "3+ Saves", "6+ Saves")]
WIN_COLS = (["F2 Win", "F2 Opponent Win"]
            + [f"F{k} {c}" for k in (3, 4, 5, 6) for c in ("Win Pred", "Opponent Win Pred")])


@pytest.fixture(scope="module")
def gw1_inputs(inputs):
    modified = dict(inputs)
    fx = inputs["fixtures"].copy()
    fx["GW2 Opponent"] = fx.iloc[:, 1]
    fx["GW2 Venue"] = fx.iloc[:, 2].map({"H": "A", "A": "H"})
    fx["GW3 Opponent"] = fx.iloc[:, 1]      # F3 fixtures so the modelled horizon is testable
    fx["GW3 Venue"] = fx.iloc[:, 2]
    modified["fixtures"] = fx
    return modified


def _build(gw1_inputs, sportsbet, roster, dc_stats, improved):
    season = team_model.season_probs(gw1_inputs)
    tv = team_model.team_fixture_view(gw1_inputs, sportsbet, draw_aware=improved)
    mkts = markets.build_all(sportsbet, gw1_inputs, dedup_f2=improved)
    return players.build(roster, season, tv, mkts, gw1_inputs["starting_lineups"],
                         gw1_inputs["fallback_factors"], dc_stats, gw1_inputs["dc_params"],
                         improved=improved)


@pytest.fixture(scope="module")
def master(gw1_inputs, sportsbet, roster, dc_stats):
    return _build(gw1_inputs, sportsbet, roster, dc_stats, improved=True)


def test_f2_fallback_engages(master):
    assert master["F2 Win"].notna().all()
    haaland = master[master["Player Name"] == "Erling Haaland"].iloc[0]
    assert haaland["F2 XP"] > 2.5  # more than appearance-only


def test_f2_pair_sums_below_one(master):
    assert ((master["F2 Win"] + master["F2 Opponent Win"]) <= 1 + 1e-9).all()


def test_all_modelled_probabilities_clamped(master):
    vals = master[[c for c in PROB_COLS + WIN_COLS if c in master.columns]]
    assert vals.min().min() >= 0.0
    assert vals.max().max() <= 1.0


def test_f2_clean_sheet_uses_factor_baseline(master):
    row = master[master["Player Name"] == "Kevin Danso"].iloc[0]
    expected = row["Clean Sheet Factor"] * model.baseline(
        "clean_sheet", pd.Series([row["F2 Win"]]), pd.Series([row["F2 Opponent Win"]]),
        pd.Series([row["Position"]]), pd.Series([row["F2 Venue"] == "H"])).iloc[0]
    assert np.isclose(row["F2 Clean Sheet"], min(max(expected, 0.0), 1.0))


def test_score_curves_are_poisson_in_improved_mode(master):
    got = master["F3 Score 2+"]
    expected = model.poisson_score2(master["F3 Score 1+"])
    pd.testing.assert_series_equal(got, expected, check_names=False)


def test_f3_score_is_blended_with_f1_odds(master):
    from fpl_pipeline import config
    w = config.PROJECTION_BLEND["score1"]
    row = master[master["Player Name"] == "Erling Haaland"].iloc[0]
    pure = row["Score 1+ Factor"] * model.baseline(
        "score1", pd.Series([row["F3 Win Pred"]]), pd.Series([row["F3 Opponent Win Pred"]]),
        pd.Series([row["Position"]]), pd.Series([row["F3 Venue"] == "H"])).iloc[0]
    expected = w * min(max(pure, 0.0), 1.0) + (1 - w) * row["F1 Score 1+"]
    assert np.isclose(row["F3 Score 1+"], expected)


def test_f2_score_uses_generic_baseline_not_sheet_model(master):
    from fpl_pipeline import config
    w = config.PROJECTION_BLEND["score1"]
    row = master[master["Player Name"] == "Erling Haaland"].iloc[0]
    pure = row["Score 1+ Factor"] * model.baseline(
        "score1", pd.Series([row["F2 Win"]]), pd.Series([row["F2 Opponent Win"]]),
        pd.Series([row["Position"]]), pd.Series([row["F2 Venue"] == "H"])).iloc[0]
    expected = w * min(max(pure, 0.0), 1.0) + (1 - w) * row["F1 Score 1+"]
    assert np.isclose(row["F2 Score 1+"], expected)


def test_unblended_stats_stay_pure_model(master):
    # yellow backtested best at w=1.0 — must remain pure factor x baseline
    row = master[master["Player Name"] == "Erling Haaland"].iloc[0]
    pure = row["F1 Yellow Card Factor"] * model.baseline(
        "yellow", pd.Series([row["F3 Win Pred"]]), pd.Series([row["F3 Opponent Win Pred"]]),
        pd.Series([row["Position"]]), pd.Series([row["F3 Venue"] == "H"])).iloc[0]
    assert np.isclose(row["F3 Yellow Card"], min(max(pure, 0.0), 1.0))


def test_odds_take_precedence_over_predictions(gw1_inputs, sportsbet, roster, dc_stats):
    sb2 = dict(sportsbet)
    wdw = sportsbet["wdw"]
    fx = gw1_inputs["fixtures"]
    gw2_home = fx[fx["GW2 Venue"] == "H"]
    extra = pd.DataFrame({
        wdw.columns[0]: gw2_home.iloc[:, 0].values,
        wdw.columns[1]: gw2_home["GW2 Opponent"].values,
        wdw.columns[2]: 2.0, wdw.columns[3]: 3.5,
    })
    sb2["wdw"] = pd.concat([wdw, extra], ignore_index=True)
    mb = _build(gw1_inputs, sb2, roster, dc_stats, improved=True)

    home_rows = mb[mb["Team"].isin(set(gw2_home.iloc[:, 0]))]["F2 Win"].dropna()
    assert np.allclose(home_rows, 1 / 2.0 / 1.03)  # no draw odds -> legacy de-margin


def test_parity_mode_uses_ladders(gw1_inputs, sportsbet, roster, dc_stats):
    mp = _build(gw1_inputs, sportsbet, roster, dc_stats, improved=False)
    got = mp["F3 Score 2+"]
    expected = model.ladder_score2(mp["F3 Score 1+"])
    pd.testing.assert_series_equal(got, expected, check_names=False)
