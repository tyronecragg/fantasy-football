import pandas as pd

from fpl_pipeline import history


def _wdw():
    return pd.DataFrame({
        "home_team": ["A", "B"], "away_team": ["C", "D"],
        "home_win_odds": [1.5, 2.0], "away_win_odds": [6.0, 3.5],
        "draw_odds": [4.0, 3.4],
    })


def _season():
    return pd.DataFrame({"team": ["A", "B", "C", "D"], "title_odds": [2.0, 10, 50, 100],
                         "relegation_odds": [500, 100, 20, 8], "top6_odds": [1.5, 3.0, 15, 40]})


def test_fixture_archive_records_unbackfillable_odds(tmp_path):
    path = tmp_path / "fixtures.csv"
    pd.DataFrame(columns=["Season", "home_team", "away_team", "home_win_odds", "away_win_odds",
                          "home_title_odds", "away_title_odds", "home_relegation_odds",
                          "away_relegation_odds", "home_top_6_odds", "away_top_6_odds"]).to_csv(path, index=False)

    sportsbet = {
        "clean_sheet": pd.DataFrame({"team_name": ["A", "B", "C", "D"],
                                     "clean_sheet_yes": [1.8, 2.5, 5.0, 4.0]}),
        "team_goals": pd.DataFrame({"Team": ["A", "B", "C", "D"],
                                    "Team_Over_1.5": [1.4, 1.9, 3.5, 2.8],
                                    "Team_Over_3.5": [3.2, 5.0, 12.0, 9.0]}),
    }
    history.update_fixture_history(_wdw(), _season(), path=str(path), season="2026-2027",
                                   gameweek=1, sportsbet=sportsbet)

    d = pd.read_csv(path)
    assert list(d["Gameweek"]) == [1, 1]                 # rows are dated
    assert list(d["draw_odds"]) == [4.0, 3.4]            # 1X2 can be de-margined properly
    assert list(d["home_clean_sheet_odds"]) == [1.8, 2.5]
    assert list(d["home_over_1.5_odds"]) == [1.4, 1.9]   # with over_3.5, pins each team lambda
    assert list(d["away_over_3.5_odds"]) == [12.0, 9.0]


def test_fixture_archive_replaces_on_rerun(tmp_path):
    path = tmp_path / "fixtures.csv"
    pd.DataFrame(columns=["Season", "home_team", "away_team", "home_win_odds", "away_win_odds",
                          "home_title_odds", "away_title_odds", "home_relegation_odds",
                          "away_relegation_odds", "home_top_6_odds", "away_top_6_odds"]).to_csv(path, index=False)
    for _ in range(2):
        history.update_fixture_history(_wdw(), _season(), path=str(path), season="2026-2027",
                                       gameweek=1, sportsbet=None)
    assert len(pd.read_csv(path)) == 2   # upsert, not append
