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

    # A: current 900min@6 + prior 1800min@10 capped at 1710 -> (900*6 + 1710*10)/2610 = 8.6207
    assert abs(dc.loc["A X", "dc90"] - (900 * 6 + 1710 * 10) / (900 + 1710)) < 1e-9
    assert abs(dc.loc["A X", "nineties"] - 29.0) < 0.01

    # B: no current data -> pure prior (capped weight), dc90 unchanged
    assert abs(dc.loc["B X", "dc90"] - 12.0) < 1e-9
    assert abs(dc.loc["B X", "nineties"] - 19.0) < 0.01

    # C: new to the league last season? no prior row -> current only
    assert abs(dc.loc["C X", "dc90"] - 4.0) < 1e-9
    assert abs(dc.loc["C X", "nineties"] - 5.0) < 0.01

    # D: no data either season -> NaN (position average downstream)
    assert pd.isna(dc.loc["D X", "dc90"])
    assert dc.loc["D X", "nineties"] == 0.0


def test_dc_shrinkage_toward_population_average():
    """Improved mode: weight = nineties/4 capped at 1 on the player's own hit-probability,
    the rest on the reliable-population mean; parity keeps the hard >=4 cliff."""
    import pandas as pd
    from fpl_pipeline import players, model, config
    prm = pd.Series({"threshold": 10, "sd": 4.27, "average_dc90": 0.44})
    dc = pd.DataFrame({"name": ["a", "b", "c", "d", "e", "f", "g"],
                       "dc90": [12.0, 8.0, 14.0, 14.0, float("nan"), 14.0, 14.0],
                       "nineties": [10.0, 10.0, 1.0, 3.0, 0.0, 0.4, 0.65]})
    out = players._dc_table(dc, prm, improved=True).set_index("name")
    avg = model.dc_probability(pd.Series([12.0, 8.0]), 4.27, 10).mean()
    own_c = model.dc_probability(pd.Series([14.0]), 4.27, 10).iloc[0]
    assert abs(out.loc["a", "prob_filled"] - out.loc["a", "prob"]) < 1e-12          # reliable: own
    assert abs(out.loc["c", "prob_filled"] - (0.25 * own_c + 0.75 * avg)) < 1e-12   # 1 ninety: 25/75
    assert abs(out.loc["d", "prob_filled"] - (0.75 * own_c + 0.25 * avg)) < 1e-12   # 3 nineties: 75/25
    assert abs(out.loc["e", "prob_filled"] - avg) < 1e-12                            # no evidence: average
    assert abs(out.loc["f", "prob_filled"] - avg) < 1e-12                            # 0.4 < 0.65: no weight
    assert abs(out.loc["g", "prob_filled"] - ((0.65 / 4) * own_c + (1 - 0.65 / 4) * avg)) < 1e-12  # 0.65: 0.65/4 own
    par = players._dc_table(dc, prm, improved=False).set_index("name")
    assert par.loc["c", "prob_filled"] == 0.44 and par.loc["d", "prob_filled"] == 0.44  # parity cliff
    assert abs(par.loc["a", "prob_filled"] - par.loc["a", "prob"]) < 1e-12
