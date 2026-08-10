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


def test_ffs_scrape_stages_without_touching_curated(tmp_path, capsys):
    pytest.importorskip("bs4")
    import starting_lineups as sl

    curated = tmp_path / "curated.csv"
    pd.DataFrame({"Player": ["Keep Me", "Judged Out"], "Team": ["T1", "T2"],
                  **{f"F{k}": [1.0, 0.5] for k in range(1, 7)}}).to_csv(curated, index=False)

    scraped = pd.DataFrame({"Player": ["Keep Me", "New Guy"], "Team": ["T1", "T3"]})
    staging = tmp_path / "staged.csv"
    sl.stage_predictions(scraped, staging_csv=str(staging), curated_csv=str(curated))

    out = capsys.readouterr().out
    assert pd.read_csv(staging)["Player"].tolist() == ["Keep Me", "New Guy"]
    kept = pd.read_csv(curated)
    assert kept["Player"].tolist() == ["Keep Me", "Judged Out"]   # curated file untouched
    assert kept["F1"].tolist() == [1.0, 0.5]
    assert "New Guy" in out and "Judged Out" in out               # diff reported for curation


def test_extract_team_news_parses_ffs_block():
    bs4 = pytest.importorskip("bs4")
    import starting_lineups as sl

    # Mirrors the observed structure of the FFS team-news page (Aug 2026)
    html = """
    <li class="team-news-item">
      <div class="story-wrap"><header><h2>Liverpool</h2></header></div>
      <div class="next-match">Next Match: Bournemouth (H)</div>
      <ul class="story-parts">
        <li class="headers"><strong>Out:</strong>
          <ul class="players"><li>Ekitik&eacute;</li></ul></li>
        <li class="headers"><strong>Doubts:</strong>
          <ul class="players">
            <li>Bradley<span class="doubt-percent">25%</span></li>
            <li>Leoni<span class="doubt-percent">50%</span></li>
          </ul></li>
        <li class="headers"><strong>Banned:</strong></li>
        <li><p><strong>Latest News:</strong> Slot confirmed Alisson trains fully.</p></li>
        <li class="headers grey"><em>Last Updated Tue 4th Aug</em></li>
      </ul>
    </li>"""
    item = bs4.BeautifulSoup(html, "html.parser").find("li", class_="team-news-item")
    news = sl.extract_team_news(item)

    assert news["next_match"] == "Bournemouth (H)"
    assert news["out"] == ["Ekitiké"]
    assert news["doubts"] == ["Bradley (25%)", "Leoni (50%)"]
    assert "banned" not in news or news["banned"] == []
    assert news["latest"] == "Slot confirmed Alisson trains fully."
    assert news["updated"] == "Last Updated Tue 4th Aug"
