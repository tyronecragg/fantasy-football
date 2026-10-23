import pandas as pd
import pytest

from fpl_pipeline import fixture_risk


def _master():
    return pd.DataFrame({"Player Name": ["A", "B", "C"], "Team": ["Liverpool", "Brighton", "Arsenal"],
                         "F1 XP": [5.0, 4.0, 6.0], "F1 XP Pre": [4.5, 3.6, 5.4], "F2 XP": [5.0, 4.0, 6.0]})


def test_load_only_current_gameweek(tmp_path):
    f = tmp_path / "risk.csv"
    f.write_text("# comment\nGameweek,Team,play_prob,note\n8,Liverpool,0.85,storm\n8,Brighton,0.85,storm\n7,Arsenal,0.5,old\n")
    assert fixture_risk.load(8, path=str(f)) == {"Liverpool": (0.85, None), "Brighton": (0.85, None)}
    assert fixture_risk.load(9, path=str(f)) == {}
    assert fixture_risk.load(8, path=str(tmp_path / "missing.csv")) == {}


def test_apply_scales_only_f1_points_of_listed_teams():
    m = _master()
    out, applied = fixture_risk.apply(m, {"Liverpool": 0.85, "Brighton": 0.85})
    assert applied == {"Liverpool": (0.85, None), "Brighton": (0.85, None)}
    assert out["F1 XP"].tolist() == pytest.approx([4.25, 3.4, 6.0])
    assert out["F1 XP Pre"].tolist() == pytest.approx([3.825, 3.06, 5.4])
    assert out["F2 XP"].tolist() == [5.0, 4.0, 6.0]                 # later weeks untouched
    assert m["F1 XP"].tolist() == [5.0, 4.0, 6.0]                   # input not mutated


def test_no_risk_is_a_no_op():
    m = _master()
    out, applied = fixture_risk.apply(m, {})
    assert out is m and applied == {}


def test_replay_week_gets_the_postponed_share(tmp_path):
    f = tmp_path / "risk.csv"
    f.write_text("Gameweek,Team,play_prob,replay_gw,note\n8,Liverpool,0.9,15,storm\n8,Brighton,0.9,,storm\n")
    risk = fixture_risk.load(8, path=str(f))
    assert risk == {"Liverpool": (0.9, 15), "Brighton": (0.9, None)}
    m = _master().assign(**{"F8 XP": [4.0, 3.0, 5.0]})
    out, applied = fixture_risk.apply(m, risk, gameweek=8)
    assert applied == {"Liverpool": (0.9, "F8 XP"), "Brighton": (0.9, None)}
    assert out["F1 XP"].tolist() == pytest.approx([4.5, 3.6, 6.0])
    assert out["F8 XP"].tolist() == pytest.approx([4.0 + 0.1 * 5.0, 3.0, 5.0])   # replay share only where given
