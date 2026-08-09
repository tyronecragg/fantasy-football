"""Minutes-weighted blending of current and prior-season defensive contribution."""
import os

import pandas as pd
import pytest

from fpl_pipeline import config, ingest


@pytest.fixture
def seasons(tmp_path, monkeypatch):
    def write_season(season, rows):
        d = tmp_path / season
        d.mkdir()
        players = pd.DataFrame([{"player_id": i, "first_name": n, "second_name": "X",
                                 "position": "Defender", "team_code": 1}
                                for i, (n, _, _) in enumerate(rows)])
        stats = pd.DataFrame([{"id": i, "gw": 1, "minutes": m, "defensive_contribution_per_90": dc}
                              for i, (_, m, dc) in enumerate(rows)])
        players.to_csv(d / "players.csv", index=False)
        stats.to_csv(d / "playerstats.csv", index=False)

    # (name, minutes, dc90) — names become "<name> X"
    write_season("2025-2026", [("A", 1800, 10.0), ("B", 2700, 12.0)])
    write_season("2026-2027", [("A", 900, 6.0), ("B", 0, None), ("C", 450, 4.0), ("D", 0, None)])
    monkeypatch.setattr(config, "SEASON", "2026-2027")
    monkeypatch.setattr(config, "FPL_DATA_DIR", str(tmp_path / "2026-2027"))
    return tmp_path


def test_minutes_weighted_blend(seasons):
    dc = ingest.load_defensive_contributions()["DEF"].set_index("name")

    # A: current 900min@6 + prior capped at 900min@10 -> (900*6 + 900*10)/1800 = 8
    assert abs(dc.loc["A X", "dc90"] - 8.0) < 1e-9
    assert abs(dc.loc["A X", "nineties"] - 20.0) < 0.01

    # B: no current data -> pure prior (capped weight), dc90 unchanged
    assert abs(dc.loc["B X", "dc90"] - 12.0) < 1e-9
    assert abs(dc.loc["B X", "nineties"] - 10.0) < 0.01

    # C: new to the league last season? no prior row -> current only
    assert abs(dc.loc["C X", "dc90"] - 4.0) < 1e-9
    assert abs(dc.loc["C X", "nineties"] - 5.0) < 0.01

    # D: no data either season -> NaN (position average downstream)
    assert pd.isna(dc.loc["D X", "dc90"])
    assert dc.loc["D X", "nineties"] == 0.0
