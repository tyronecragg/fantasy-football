import pandas as pd

from fpl_pipeline import reconcile


def _roster():
    return pd.DataFrame({"name": ["Josh King", "Erling Haaland", "Sven Botman"],
                         "team": ["Fulham", "Man City", "Newcastle"],
                         "position": ["MID", "FWD", "DEF"]})


def _lineups(players):
    return pd.DataFrame({"Player": players, "Team": ["Fulham"] * len(players),
                         "F1": [1.0] * len(players)})


def _mkts(score_players, assist_players=()):
    return {"score1": pd.DataFrame({"player": list(score_players)}),
            "assist": pd.DataFrame({"player": list(assist_players)})}


def test_clean_sources_report_nothing():
    rec = reconcile.report(_roster(), _lineups(["Josh King"]), _mkts(["Josh King"]))
    assert rec.empty


def test_lineup_mismatch_detected_with_suggestion():
    rec = reconcile.report(_roster(), _lineups(["Joshua King"]), _mkts(["Joshua King"]))
    row = rec[rec["source"] == "starting_lineups"].iloc[0]
    assert row["name"] == "Joshua King"
    assert row["suggestion"] == "Josh King"      # unique-surname match


def test_odds_accent_variant_suggested():
    rec = reconcile.report(_roster(), _lineups(["Josh King"]),
                           _mkts(["Josh King", "Erling Håland"]))
    row = rec[rec["source"] == "odds:score1"].iloc[0]
    assert row["suggestion"] == "Erling Haaland"


def test_combo_markets_ignored():
    rec = reconcile.report(_roster(), _lineups(["Josh King"]),
                           _mkts(["Josh King"], ["Josh King or Erling Haaland to assist"]))
    assert (rec["source"] != "odds:assist").all()


def test_starter_without_attacking_odds_flagged():
    rec = reconcile.report(_roster(), _lineups(["Josh King"]), _mkts([], []))
    assert (rec["source"] == "coverage").any()
    assert "Josh King" in set(rec[rec["source"] == "coverage"]["name"])


def _one(name, team, position):
    return pd.DataFrame({"name": [name], "team": [team], "position": [position]})


def test_any_token_abbreviation_matches_non_final_surname():
    # 'Guimaraes' is token 2 of 5, not the last token — the old last-token rule missed this.
    roster = _one("Bruno Guimaraes Rodrigues da Silva", "Newcastle", "MID")
    lin = pd.DataFrame({"Player": ["Bruno Guimaraes"], "Team": ["Newcastle"], "F1": [1.0]})
    rec = reconcile.report(roster, lin, _mkts([], []))
    row = rec[rec["source"] == "starting_lineups"].iloc[0]
    assert row["suggestion"] == "Bruno Guimaraes Rodrigues da Silva"


def test_surname_collision_still_refused():
    # Same surname, different player: must NOT propose the forward onto the defender.
    roster = _one("Matheus Cunha", "Man Utd", "FWD")
    lin = pd.DataFrame({"Player": ["Jair Cunha"], "Team": ["Nott'm Forest"], "F1": [1.0]})
    rec = reconcile.report(roster, lin, _mkts([], []))
    row = rec[rec["name"] == "Jair Cunha"].iloc[0]
    assert row["suggestion"] != "Matheus Cunha"


def test_shared_first_name_alone_does_not_match():
    roster = _one("James Maddison", "Spurs", "MID")
    lin = pd.DataFrame({"Player": ["James Rodriguez"], "Team": ["Spurs"], "F1": [1.0]})
    rec = reconcile.report(roster, lin, _mkts([], []))
    row = rec[rec["name"] == "James Rodriguez"].iloc[0]
    assert row["suggestion"] != "James Maddison"


def test_read_csv_tolerant_repairs_excel_ansi_save(tmp_path):
    """An Excel ANSI re-save mixes plain cp1252 accents with double-encoded UTF-8
    ('Touré' displayed and saved as 'TourÃ©'). The tolerant reader must repair both
    and heal the file to clean UTF-8."""
    from fpl_pipeline.io_utils import read_csv_tolerant

    path = tmp_path / "lineups.csv"
    mojibake = "Touré".encode("utf-8").decode("cp1252")   # what Excel displayed
    raw = ("Player\n" + mojibake + "\nMartínez\n").encode("cp1252")  # what Excel saved
    path.write_bytes(raw)

    df = read_csv_tolerant(str(path))
    assert df["Player"].tolist() == ["Touré", "Martínez"]
    # file healed: plain UTF-8 read now works
    import pandas as pd
    assert pd.read_csv(str(path))["Player"].tolist() == ["Touré", "Martínez"]
