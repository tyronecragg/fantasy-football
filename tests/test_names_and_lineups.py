import pandas as pd
import pytest

from fpl_pipeline import names


def test_mapping_table_integrity():
    df = pd.read_csv(names.NAME_MAPPINGS_CSV)
    assert set(df.columns) == {"type", "name", "name_cleaned"}
    assert set(df["type"]) <= {"player", "team"}
    dupes = df[df.duplicated(subset=["type", "name"], keep=False)]
    assert dupes.empty, f"duplicate mapping keys:\n{dupes}"


def test_known_mappings_applied():
    s = pd.Series(["David Raya Martin", "Unknown Player"])
    out = names.apply_player_names(s)
    assert out.tolist() == ["David Raya", "Unknown Player"]
    t = pd.Series(["Tottenham", "Nottingham Forest", "Arsenal"])
    assert names.apply_team_names(t).tolist() == ["Spurs", "Nott'm Forest", "Arsenal"]


def test_lineup_update_preserves_start_probs(tmp_path):
    pytest.importorskip("bs4")
    import starting_lineups as sl

    path = str(tmp_path / "lineups.csv")
    pd.DataFrame({"Player": ["Keep Me", "Drop Me"], "Team": ["T1", "T2"],
                  "F1": [1.0, 0.5], "F2": [0.75, 0.5], "F3": [1, 1],
                  "F4": [1, 1], "F5": [1, 1], "F6": [1, 1]}).to_csv(path, index=False)

    scraped = pd.DataFrame({"Player": ["Keep Me", "New Guy"], "Team": ["T1", "T3"]})
    sl.update_inputs_csv(scraped, inputs_csv=path)

    out = pd.read_csv(path).set_index("Player")
    assert list(out.index) == ["Keep Me", "New Guy"]
    assert out.loc["Keep Me", "F1"] == 1.0 and out.loc["Keep Me", "F2"] == 0.75
    assert out.loc["New Guy", sl.PROB_COLUMNS].isna().all()
