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
    monkeypatch.setattr(config, "DC_NEUTRALISE_FIXTURE_MIX", False)   # no football-data in the fixture
    return tmp_path


def test_favourite_mix_is_the_teams_average_multiplier():
    """A team that is a 0.85 favourite in one match and 0.40 in the other averages (0.82 + 1.0) / 2;
    its opponents (always underdogs) average 1.0. Dividing a rate by the mix and multiplying by the same
    fixture's multiplier returns the rate - the neutralisation cannot double-count."""
    m = pd.DataFrame({"season": ["S", "S"], "HomeTeam": ["Fav", "Dog2"], "AwayTeam": ["Dog1", "Fav"],
                      "p_home": [0.85, 0.35], "p_away": [0.05, 0.40]})
    mix = ingest.favourite_mix(m, curve={"DEF": (0.55, 0.85, 0.82)})
    assert abs(mix[("S", "Fav", "DEF")] - 0.91) < 1e-9
    assert mix[("S", "Dog1", "DEF")] == 1.0 and mix[("S", "Dog2", "DEF")] == 1.0
    assert ingest.favourite_mix(m, curve={}) == {}


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


def test_minutes_per_start_is_shrunk_and_scales_the_rate(seasons, monkeypatch):
    """A's starts: 4 this season at 70 min (recency-weighted), 20 last season at 90 (capped at 10).
    Shrunk toward the DEF average with k pseudo-starts. B has no starts anywhere -> the position average."""
    from fpl_pipeline import players, model
    for season, rows in [("2026-2027", [(0, 4, 70)]), ("2025-2026", [(0, 20, 90)])]:
        d = seasons / season / "By Gameweek" / "GW1"
        d.mkdir(parents=True)
        recs = []
        for pid, n, mins in rows:
            recs += [{"id": pid, "gw": i + 1, "minutes": mins, "starts": 1} for i in range(n)]
        pd.DataFrame(recs).to_csv(d / "player_gameweek_stats.csv", index=False)
    dc = ingest.load_defensive_contributions()["DEF"].set_index("name")
    k, cap, avg = config.DC_MINUTES_SHRINK_STARTS, config.DC_MINUTES_PRIOR_CAP_STARTS, config.DC_MINUTES_POSITION_AVG["DEF"]
    W = config.DC_CURRENT_SEASON_WEIGHT                      # this season's starts count W x, like the rate
    assert abs(dc.loc["A X", "mps"] - (W * 4 * 70 + cap * 90 + k * avg) / (W * 4 + cap + k)) < 0.06
    assert abs(dc.loc["B X", "mps"] - avg) < 1e-9
    # full-90 share: A has 0 of 4 full this season, 10 (capped) of 20 full last season, k pseudo-starts
    avg90 = config.DC_FULL90_POSITION_AVG["DEF"]
    assert abs(dc.loc["A X", "p90"] - (W * 0 + cap * 1.0 + k * avg90) / (W * 4 + cap + k)) < 1e-3
    assert abs(dc.loc["B X", "p90"] - avg90) < 1e-3
    # the table prices a start as a mixture: p90 x P(hit | rate) + (1 - p90) x P(hit | rate x sub / 90)
    prm = pd.Series({"threshold": 10, "sd": 3.86, "average_dc90": 0.44}, name="DEF")
    tab = pd.DataFrame({"name": ["p", "q"], "dc90": [12.0, 12.0], "nineties": [10.0, 10.0], "p90": [1.0, 0.5]})
    out = players._dc_table(tab, prm, improved=True).set_index("name")
    P = lambda mu: model.dc_probability(pd.Series([mu]), 3.86, 10).iloc[0]
    sub = config.DC_SUB_MINUTES["DEF"]
    assert abs(out.loc["p", "prob_filled"] - P(12.0)) < 1e-9
    assert abs(out.loc["q", "prob_filled"] - (0.5 * P(12.0) + 0.5 * P(12.0 * sub / 90))) < 1e-9
    monkeypatch.setattr(config, "DC_MINUTES_PER_START", False)
    off = players._dc_table(tab, prm, improved=True).set_index("name")
    assert abs(off.loc["q", "prob_filled"] - P(12.0)) < 1e-9


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
