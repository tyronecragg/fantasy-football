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
        pd.DataFrame([{"code": 1, "name": "T"}]).to_csv(d / "teams.csv", index=False)
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

    # A: current 900min@6 weighted W x + prior 1800min@10 capped at 1710. The RATE tilts to the
    # recent (lower) form via W; the EVIDENCE count (nineties) stays TRUE minutes (900+1710)/90=29,
    # undoubled — proving the two denominators are kept separate.
    W = config.DC_CURRENT_SEASON_WEIGHT
    assert abs(dc.loc["A X", "dc90"] - (W * 900 * 6 + 1710 * 10) / (W * 900 + 1710)) < 1e-9
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


def test_external_prior_fills_gaps_only(tmp_path, monkeypatch):
    """External DefCon prior backfills players with NO FPL history; never overrides one who has it."""
    ext = pd.DataFrame({"name": ["new_signing", "has_pl_history"], "team": ["Hull City", "Arsenal"],
                        "position": ["DEF", "MID"], "dc90": [9.0, 99.0], "minutes": [1710, 1710],
                        "source": ["championship_2025_26", "championship_2025_26"]})
    path = tmp_path / "external_dc_prior.csv"
    ext.to_csv(path, index=False)
    monkeypatch.setattr(config, "EXTERNAL_DC_PRIOR", str(path))
    fpl_prior = pd.DataFrame({"minutes": [1000.0], "team": ["Arsenal"], "dc90": [5.0]},
                             index=pd.Index(["has_pl_history"], name="name"))
    merged = ingest._merge_external_prior(fpl_prior)
    assert merged.loc["new_signing", "dc90"] == 9.0            # gap filled from external
    assert merged.loc["has_pl_history", "dc90"] == 5.0         # FPL history wins, external 99.0 ignored


def test_external_prior_absent_is_noop(monkeypatch):
    monkeypatch.setattr(config, "EXTERNAL_DC_PRIOR", "/nonexistent/path.csv")
    fpl_prior = pd.DataFrame({"minutes": [1000.0], "team": ["Arsenal"], "dc90": [5.0]},
                             index=pd.Index(["x"], name="name"))
    assert ingest._merge_external_prior(fpl_prior).equals(fpl_prior)


def test_dc_shrinkage_toward_population_average():
    """Improved mode: the RATE (dc90) is shrunk toward the reliable-population mean dc90 by
    weight = min(nineties/DC_SHRINK_NINETIES, 1) (zeroed below DC_SHRINK_MIN_NINETIES), THEN
    converted to a probability. Assertions replicate the formula so they track any gate.
    Parity keeps the hard >=4 cliff on the probability."""
    import pandas as pd
    from fpl_pipeline import players, model, config
    g, mn = config.DC_SHRINK_NINETIES, config.DC_SHRINK_MIN_NINETIES
    prm = pd.Series({"threshold": 10, "sd": 4.27, "average_dc90": 0.44})
    dc = pd.DataFrame({"name": ["a", "b", "c", "d", "e", "f", "gg"],
                       "dc90": [12.0, 8.0, 14.0, 14.0, float("nan"), 14.0, 14.0],
                       "nineties": [10.0, 10.0, 1.0, 3.0, 0.0, 0.4, 0.65]})
    out = players._dc_table(dc, prm, improved=True).set_index("name")
    P = lambda rate: model.dc_probability(pd.Series([rate]), 4.27, 10).iloc[0]
    avg = dc.loc[dc["nineties"] >= g, "dc90"].mean()        # mean dc90 of reliable, same as the code
    for _, r in dc.iterrows():
        n = r["nineties"]
        w = 0.0 if n < mn else min(n / g, 1.0)
        rate = w * (r["dc90"] if pd.notna(r["dc90"]) else avg) + (1 - w) * avg
        assert abs(out.loc[r["name"], "prob_filled"] - P(rate)) < 1e-9   # rate-space blend, gate-agnostic
    par = players._dc_table(dc, prm, improved=False).set_index("name")
    assert par.loc["c", "prob_filled"] == 0.44 and par.loc["d", "prob_filled"] == 0.44  # parity cliff (>=4)
    assert abs(par.loc["a", "prob_filled"] - par.loc["a", "prob"]) < 1e-12
