"""History upserts: replace-on-rerun, never duplicate."""
import pandas as pd

from fpl_pipeline import history


def test_player_history_replaces_same_season_gameweek(tmp_path):
    path = str(tmp_path / "hist.csv")
    pd.DataFrame({"Season": ["2025-2026"] * 2, "Gameweek": [1, 1],
                  "Player Name": ["A", "B"], "Cost": [5.0, 6.0]}).to_csv(path, index=False)
    master = pd.DataFrame({"Player Name": ["A", "B"], "Cost": [5.5, 6.5]})

    out = history.update_player_history(master, gameweek=2, path=path, season="2025-2026")
    assert len(out) == 4  # appended new gameweek

    master2 = pd.DataFrame({"Player Name": ["A", "B"], "Cost": [9.9, 9.9]})
    out = history.update_player_history(master2, gameweek=2, path=path, season="2025-2026")
    assert len(out) == 4  # rerun replaced, not appended
    gw2 = out[out["Gameweek"] == 2]
    assert (gw2["Cost"] == 9.9).all()
    assert (out[out["Gameweek"] == 1]["Cost"] == [5.0, 6.0]).all()  # other GWs untouched


def test_player_history_gameweeks_isolated_by_season(tmp_path):
    path = str(tmp_path / "hist.csv")
    pd.DataFrame({"Season": ["2025-2026"] * 2, "Gameweek": [16, 16],
                  "Player Name": ["A", "B"], "Cost": [5.0, 6.0]}).to_csv(path, index=False)

    master = pd.DataFrame({"Player Name": ["A", "B"], "Cost": [7.0, 8.0]})
    out = history.update_player_history(master, gameweek=16, path=path, season="2026-2027")

    assert len(out) == 4  # same GW number, different season -> appended, NOT replaced
    old = out[out["Season"] == "2025-2026"]
    assert (old["Cost"] == [5.0, 6.0]).all()


def test_fixture_history_upserts_by_season_and_pair(tmp_path):
    path = str(tmp_path / "fixtures.csv")
    cols = ["Season", "home_team", "away_team", "home_win_odds", "away_win_odds",
            "home_title_odds", "away_title_odds", "home_relegation_odds",
            "away_relegation_odds", "home_top_6_odds", "away_top_6_odds"]
    pd.DataFrame([["2025-2026", "Alpha", "Beta", 2.0, 3.0, 10, 20, 100, 200, 5, 6]],
                 columns=cols).to_csv(path, index=False)

    wdw = pd.DataFrame({"home_team": ["Alpha"], "away_team": ["Beta"],
                        "home_win_odds": [1.8], "away_win_odds": [4.2]})
    season = pd.DataFrame({"team": ["Alpha", "Beta"], "title_odds": [11.0, 21.0],
                           "relegation_odds": [101.0, 201.0], "top6_odds": [7.0, 8.0]})

    # Same fixture pair in a NEW season: appended, old season's row untouched
    out = history.update_fixture_history(wdw, season, path=path, season="2026-2027")
    assert len(out) == 2

    # Rerun within the same season: replaced
    wdw2 = wdw.assign(home_win_odds=[1.5])
    out = history.update_fixture_history(wdw2, season, path=path, season="2026-2027")
    assert len(out) == 2
    new_row = out[out["Season"] == "2026-2027"].iloc[0]
    assert new_row["home_win_odds"] == 1.5
    assert out[out["Season"] == "2025-2026"].iloc[0]["home_win_odds"] == 2.0


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


def test_season_weekly_factors(tmp_path):
    path = str(tmp_path / "hist.csv")
    rows = []
    for gw, prob in ((16, 0.6), (17, 0.5), (18, 0.66)):
        rows.append({"Season": "2025-2026", "Gameweek": gw, "Player Name": "A",
                     "Position": "FWD", "F1 Venue": "H", "F1 Win": 0.5,
                     "F1 Opponent Win": 0.3, "F1 Score 1+": prob})
    rows.append({"Season": "2024-2025", "Gameweek": 16, "Player Name": "A",
                 "Position": "FWD", "F1 Venue": "H", "F1 Win": 0.5,
                 "F1 Opponent Win": 0.3, "F1 Score 1+": 0.99})  # other season: excluded
    pd.DataFrame(rows).to_csv(path, index=False)

    out = history.season_weekly_factors("score1", season="2025-2026", path=path)
    assert set(out) == {"A"} and len(out["A"]) == 3
    assert history.season_weekly_factors("score1", season="2026-2027", path=path) == {}
