"""History upserts: replace-on-rerun, never duplicate."""
import pandas as pd

from fpl_pipeline import history


def test_player_history_replaces_same_gameweek(tmp_path):
    path = str(tmp_path / "hist.csv")
    pd.DataFrame({"Gameweek": [1, 1], "Player Name": ["A", "B"], "Cost": [5.0, 6.0]}).to_csv(path, index=False)
    master = pd.DataFrame({"Player Name": ["A", "B"], "Cost": [5.5, 6.5]})

    out = history.update_player_history(master, gameweek=2, path=path)
    assert len(out) == 4  # appended new gameweek

    master2 = pd.DataFrame({"Player Name": ["A", "B"], "Cost": [9.9, 9.9]})
    out = history.update_player_history(master2, gameweek=2, path=path)
    assert len(out) == 4  # rerun replaced, not appended
    gw2 = out[out["Gameweek"] == 2]
    assert (gw2["Cost"] == 9.9).all()
    assert (out[out["Gameweek"] == 1]["Cost"] == [5.0, 6.0]).all()  # other GWs untouched


def test_fixture_history_upserts_by_pair(tmp_path):
    path = str(tmp_path / "fixtures.csv")
    cols = ["home_team", "away_team", "home_win_odds", "away_win_odds",
            "home_title_odds", "away_title_odds", "home_relegation_odds",
            "away_relegation_odds", "home_top_6_odds", "away_top_6_odds"]
    pd.DataFrame([["Old A", "Old B", 2.0, 3.0, 10, 20, 100, 200, 5, 6]], columns=cols).to_csv(path, index=False)

    wdw = pd.DataFrame({"home_team": ["Alpha"], "away_team": ["Beta"],
                        "home_win_odds": [1.8], "away_win_odds": [4.2]})
    season = pd.DataFrame({"team": ["Alpha", "Beta"], "title_odds": [11.0, 21.0],
                           "relegation_odds": [101.0, 201.0], "top6_odds": [7.0, 8.0]})

    out = history.update_fixture_history(wdw, season, path=path)
    assert len(out) == 2  # old pair kept, new pair appended

    wdw2 = wdw.assign(home_win_odds=[1.5])
    out = history.update_fixture_history(wdw2, season, path=path)
    assert len(out) == 2  # same pair replaced
    row = out[out["home_team"] == "Alpha"].iloc[0]
    assert row["home_win_odds"] == 1.5
    assert row["away_title_odds"] == 21.0


def test_fallback_factors_only_overwrite_with_real_values(tmp_path):
    path = str(tmp_path / "fallback.csv")
    cols = ["Player Name"] + history.FACTOR_COLUMNS
    pd.DataFrame([["A"] + [1.0] * 7, ["B"] + [2.0] * 7], columns=cols).to_csv(path, index=False)

    master = pd.DataFrame([["A", 0.9] + [3.0] + [None] * 6, ["C", 0.8] + [4.0] * 7],
                          columns=["Player Name", "F1 Win"] + history.FACTOR_COLUMNS)
    out = history.refresh_fallback_factors(master, path=path).set_index("Player Name")

    assert out.loc["A", history.FACTOR_COLUMNS[0]] == 3.0   # fresh value overwrites
    assert out.loc["A", history.FACTOR_COLUMNS[1]] == 1.0   # NaN does NOT overwrite
    assert out.loc["B", history.FACTOR_COLUMNS[0]] == 2.0   # absent player untouched
    assert out.loc["C", history.FACTOR_COLUMNS[0]] == 4.0   # new player appended
