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


def test_player_history_withholds_synthetic_market_columns(tmp_path):
    path = tmp_path / "hist.csv"
    pd.DataFrame(columns=["Season", "Gameweek", "Player Name", "F1 Yellow Card",
                          "F1 Yellow Card Factor", "F2 Yellow Card", "F1 Score 1+"]).to_csv(path, index=False)
    master = pd.DataFrame({"Player Name": ["A"], "F1 Yellow Card": [0.3], "F1 Yellow Card Factor": [1.1],
                           "F2 Yellow Card": [0.25], "F1 Score 1+": [0.4]})
    out = history.update_player_history(master, gameweek=3, path=path, season="2026-2027",
                                        withhold=("Yellow Card",))
    row = out.iloc[0]
    assert pd.isna(row["F1 Yellow Card"]) and pd.isna(row["F1 Yellow Card Factor"]) and pd.isna(row["F2 Yellow Card"])
    assert row["F1 Score 1+"] == 0.4


def test_fixture_history_refuses_a_stale_wdw_block(tmp_path):
    path = tmp_path / "fix.csv"
    pd.DataFrame(columns=["Season", "Gameweek", "home_team", "away_team", "home_win_odds",
                          "away_win_odds"]).to_csv(path, index=False)
    wdw = pd.DataFrame({"home_team": ["A", "C"], "away_team": ["B", "D"],
                        "home_win_odds": [2.0, 2.0], "away_win_odds": [3.0, 3.0]})
    season_probs = pd.DataFrame({"team": list("ABCD"), "title_odds": 10.0,
                                 "relegation_odds": 10.0, "top6_odds": 10.0})
    # fixtures.csv says GW5 is A-C / B-D, but the W/D/W block is A-B / C-D (last week's) -> refuse
    fixtures = pd.DataFrame({"Team": list("ABCD"), "GW5 Opponent": ["C", "D", "A", "B"]})
    out = history.update_fixture_history(wdw, season_probs, path=path, season="2026-2027",
                                         gameweek=5, fixtures=fixtures)
    assert out.empty
    fixtures = pd.DataFrame({"Team": list("ABCD"), "GW5 Opponent": ["B", "A", "D", "C"]})
    out = history.update_fixture_history(wdw, season_probs, path=path, season="2026-2027",
                                         gameweek=5, fixtures=fixtures)
    assert len(out) == 2 and set(out["Gameweek"]) == {5}


def test_season_weekly_factors_ignore_zero_factors(tmp_path):
    path = tmp_path / "hist.csv"
    pd.DataFrame({"Season": ["2026-2027"] * 2, "Gameweek": [1, 2], "Player Name": ["A", "A"],
                  "Position": ["DEF", "DEF"], "F1 Venue": ["H", "H"], "F1 Win": [0.5, 0.5],
                  "F1 Opponent Win": [0.25, 0.25],
                  "F1 Concede 4+ Goals": [0.0, 0.08]}).to_csv(path, index=False)
    out = history.season_weekly_factors("concede4", season="2026-2027", path=path)
    assert len(out["A"]) == 1 and out["A"][0] > 0


def _score_archive(path, rows):
    """Archive rows (season, gw, player, position, P(score)) at a fixed fixture; returns the
    factor that one unit of probability maps to, so expected medians can be computed."""
    from fpl_pipeline import model
    recs = [{"Season": s, "Gameweek": g, "Player Name": p, "Position": pos, "F1 Venue": "H",
             "F1 Win": 0.5, "F1 Opponent Win": 0.3, "F1 Score 1+": prob} for s, g, p, pos, prob in rows]
    pd.DataFrame(recs).to_csv(path, index=False)
    base = {pos: float(model.baseline("score1", pd.Series([0.5]), pd.Series([0.3]),
                                      pd.Series([pos]), pd.Series([True]))[0]) for pos in ("MID", "FWD")}
    return base


def test_fallback_rebuilt_from_archive_median(tmp_path):
    fb, hist = str(tmp_path / "fallback.csv"), str(tmp_path / "hist.csv")
    cols = ["Player Name"] + history.FACTOR_COLUMNS
    # A: stale placeholder in the sheet; B: plausible value, no archive; Z: placeholder, no archive
    pd.DataFrame([["A"] + [0.01] * 7, ["B"] + [0.7] * 7, ["Z"] + [0.01] * 7],
                 columns=cols).to_csv(fb, index=False)
    base = _score_archive(hist, [("2025-2026", 30, "A", "MID", 0.30),
                                 ("2026-2027", 1, "A", "MID", 0.10),
                                 ("2026-2027", 2, "A", "MID", 0.20),
                                 ("2026-2027", 1, "C", "FWD", 0.40)])
    master = pd.DataFrame({"Player Name": ["A", "B", "C", "Z"], "Position": ["MID", "MID", "FWD", "MID"]})
    out = history.refresh_fallback_factors(master, path=fb, history_path=hist).set_index("Player Name")

    sc = "Score 1+ Factor"
    assert abs(out.loc["A", sc] - 0.20 / base["MID"]) < 1e-9   # median of A's real weeks, across seasons
    assert out.loc["B", sc] == 0.7                             # no history, plausible -> kept
    assert abs(out.loc["C", sc] - 0.40 / base["FWD"]) < 1e-9   # new player from the archive appended
    assert out.loc["Z", sc] >= history.FALLBACK_MIN_FACTOR     # placeholder, no history -> position median
    vals = out[history.FACTOR_COLUMNS]
    assert ((vals >= history.FALLBACK_MIN_FACTOR) | vals.isna()).all().all()   # no placeholder survives


def test_fallback_uses_only_most_recent_weeks(tmp_path):
    fb, hist = str(tmp_path / "fallback.csv"), str(tmp_path / "hist.csv")
    pd.DataFrame([["A"] + [1.0] * 7], columns=["Player Name"] + history.FACTOR_COLUMNS).to_csv(fb, index=False)
    rows = [("2025-2026", g, "A", "MID", 0.05) for g in range(1, 21)]          # 20 old low weeks
    rows += [("2026-2027", g, "A", "MID", 0.30) for g in range(1, 4)]         # 3 recent high weeks
    base = _score_archive(hist, rows)
    out = history.refresh_fallback_factors(None, path=fb, history_path=hist, weeks=3).set_index("Player Name")
    assert abs(out.loc["A", "Score 1+ Factor"] - 0.30 / base["MID"]) < 1e-9


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
