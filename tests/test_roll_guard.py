import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import roll_gameweek  # noqa: E402


def test_current_f1_gw_reads_first_fixture_column(tmp_path):
    p = tmp_path / "fixtures.csv"
    p.write_text("Team,GW7 Opponent,GW7 Venue,GW8 Opponent\nArsenal,Forest,A,Everton\n", encoding="utf-8")
    assert roll_gameweek.current_f1_gw(str(p)) == 7


def test_current_f1_gw_unreadable_is_none(tmp_path):
    assert roll_gameweek.current_f1_gw(str(tmp_path / "missing.csv")) is None
    p = tmp_path / "bad.csv"
    p.write_text("Team,Opponent\n", encoding="utf-8")
    assert roll_gameweek.current_f1_gw(str(p)) is None


def test_double_roll_refused(monkeypatch):
    monkeypatch.setattr(roll_gameweek, "current_f1_gw", lambda: 7)
    monkeypatch.setattr(roll_gameweek, "step", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran a step")))
    monkeypatch.setattr(sys, "argv", ["roll_gameweek.py", "--gw", "7"])
    try:
        roll_gameweek.main()
    except SystemExit as e:
        assert "already" in str(e)
    else:
        raise AssertionError("double roll was not refused")
